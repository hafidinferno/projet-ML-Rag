"""
Agent service for the fraud assistant.
Handles LLM calls, response parsing, and validation.
"""

import json
import re
import uuid
import time
from typing import Optional, Dict, Any, List, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models.requests import ChatRequest
from app.models.responses import AgentResponse, ChatResponse, Citation, RiskFlag, TrustLevel
from app.services.retrieval import get_retriever, RetrievedPassage
from app.prompts.system_prompt import get_system_prompt
from app.prompts.templates import build_user_message, build_query_for_retrieval
from app.utils.validators import validate_user_input
from app.utils.logging_config import get_logger, agent_logger

logger = get_logger("agent")


_STOCK_APOLOGY_PATTERNS = [
    r"^\s*Je\s+suis\s+d[ée]sol[ée]\s+pour\s+ce\s+que\s+vous\s+avez\s+v[ée]cu[, ]*\s*",
    r"^\s*D[ée]sol[ée]\s+pour\s+ce\s+que\s+vous\s+avez\s+v[ée]cu[, ]*\s*",
]


def _strip_stock_apology(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _STOCK_APOLOGY_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out.strip()


def _contains_sensitive_data(text: str) -> bool:
    if not text:
        return False
    t = text.lower()

    # PAN-like long digit sequences
    if re.search(r"\b\d{12,19}\b", t):
        return True

    # Explicit keywords
    keywords = ["cvv", "cvc", "cryptogramme", "pin", "otp", "code sms", "mot de passe", "password"]
    if any(k in t for k in keywords):
        return True

    return False


def _looks_off_topic(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    fraud_keywords = [
        "fraude", "arnaque", "phishing", "paiement", "transaction", "carte", "cb",
        "opposition", "contestation", "remboursement", "virement", "iban", "prélèvement",
        "prelevement", "sepa", "vol", "perte", "non autorisé", "non autorise"
    ]
    return not any(k in t for k in fraud_keywords)


def _best_passage_match(passages: List[RetrievedPassage], doc_id: str, page_or_section: str, excerpt: str) -> Optional[RetrievedPassage]:
    """
    Find the best matching passage for a parsed citation.
    Strategy:
      1) filter by doc_id if provided
      2) boost exact page_or_section match
      3) lexical overlap between excerpt and passage.content
      4) fallback to highest score
    """
    if not passages:
        return None

    candidates = passages
    if doc_id:
        candidates = [p for p in passages if p.doc_id == doc_id] or passages

    ex = (excerpt or "").lower()
    page = (page_or_section or "").strip()

    best: Tuple[float, RetrievedPassage] | None = None

    for p in candidates:
        s = float(p.score)

        if page and (p.page_or_section or "").strip() == page:
            s += 0.5

        if ex:
            # Overlap score
            tokens = set(re.findall(r"\b\w{3,}\b", ex))
            if tokens:
                content_tokens = set(re.findall(r"\b\w{3,}\b", (p.content or "").lower()))
                overlap = len(tokens & content_tokens) / max(1, len(tokens))
                s += overlap

        if best is None or s > best[0]:
            best = (s, p)

    return best[1] if best else None


class FraudAssistantAgent:
    """
    Main agent for fraud assistance.
    Combines RAG retrieval with Mistral LLM via Ollama.
    """

    # Below this, treat retrieval as "not enough"
    MIN_PASSAGE_SCORE = 0.12

    def __init__(self):
        self.system_prompt = get_system_prompt()
        self.http_client = httpx.Client(timeout=settings.ollama_timeout)

    def __del__(self):
        if hasattr(self, "http_client"):
            self.http_client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 4900
            }
        }

        response = self.http_client.post(settings.ollama_chat_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")

    def _parse_json_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        for match in re.findall(json_pattern, raw_response):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        brace_pattern = r"\{[\s\S]*\}"
        for match in re.findall(brace_pattern, raw_response):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        logger.warning("json_parse_failed", raw_length=len(raw_response))
        return None

    def _dedup_citations(self, citations: List[Citation]) -> List[Citation]:
        seen = set()
        out = []
        for c in citations:
            key = (c.chunk_id, c.doc_id, c.page_or_section)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _validate_and_build_response(self, parsed: Dict[str, Any], passages: List[RetrievedPassage]) -> AgentResponse:
        parsed_citations = parsed.get("citations", []) or []
        citations: List[Citation] = []

        for i, cit in enumerate(parsed_citations):
            if not isinstance(cit, dict):
                continue

            doc_id = (cit.get("doc_id") or "").strip()
            page_or_section = (cit.get("page_or_section") or "").strip()
            excerpt = (cit.get("excerpt") or "")[:200]

            passage = _best_passage_match(passages, doc_id, page_or_section, excerpt)

            citations.append(Citation(
                chunk_id=passage.chunk_id if passage else f"parsed_{i}",
                doc_id=passage.doc_id if passage else (doc_id or f"unknown_{i}"),
                title=passage.title if passage else (cit.get("title") or (doc_id or f"unknown_{i}")),
                page_or_section=passage.page_or_section if passage else (page_or_section or "N/A"),
                excerpt=excerpt or ((passage.content[:200]) if passage else ""),
                score=float(passage.score) if passage else 0.0,
                source_path=passage.source_path if passage else "",
                trust_level=TrustLevel(passage.trust_level) if passage else TrustLevel.TRUSTED
            ))

        citations = self._dedup_citations(citations)

        risk_flags: List[RiskFlag] = []
        for flag in (parsed.get("risk_flags", []) or []):
            if isinstance(flag, dict):
                risk_flags.append(RiskFlag(
                    flag_type=flag.get("flag_type", "unknown"),
                    description=flag.get("description", ""),
                    severity=flag.get("severity", "medium")
                ))

        customer_message = parsed.get("customer_message") or ""
        customer_message = _strip_stock_apology(customer_message)

        actions = parsed.get("actions", []) or []
        missing = parsed.get("missing_info_questions", []) or []
        info_not_found = bool(parsed.get("info_not_found", False))

        # Enforce: if actions exist but no citations and we had passages, attach top passages as citations
        if actions and not citations and passages:
            top = [p for p in passages if p.score >= self.MIN_PASSAGE_SCORE][:2]
            for p in top:
                citations.append(Citation(
                    chunk_id=p.chunk_id,
                    doc_id=p.doc_id,
                    title=p.title,
                    page_or_section=p.page_or_section,
                    excerpt=(p.content[:200] + "...") if len(p.content) > 200 else p.content,
                    score=float(p.score),
                    source_path=p.source_path,
                    trust_level=TrustLevel(p.trust_level)
                ))
            citations = self._dedup_citations(citations)

        # If still no citations for factual-looking answer, mark not found
        if (actions or customer_message) and not citations and not info_not_found:
            info_not_found = True
            if not missing:
                missing = ["Pouvez-vous préciser le type d’opération (paiement carte, virement, prélèvement) et le canal (en ligne / terminal) ?"]

        return AgentResponse(
            customer_message=customer_message or "Pouvez-vous préciser votre situation (paiement carte, virement, prélèvement) ?",
            actions=actions,
            missing_info_questions=missing,
            citations=citations,
            risk_flags=risk_flags,
            raw_passages_used=len(passages),
            info_not_found=info_not_found
        )

    def _build_safe_refusal_response(self, session_id: str, reason: str, retrieval_hint: str) -> AgentResponse:
        retriever = get_retriever()
        passages = retriever.retrieve(query=retrieval_hint, session_id=session_id)

        # Use best 2 citations if possible
        citations: List[Citation] = []
        for p in passages[:2]:
            citations.append(Citation(
                chunk_id=p.chunk_id,
                doc_id=p.doc_id,
                title=p.title,
                page_or_section=p.page_or_section,
                excerpt=(p.content[:200] + "...") if len(p.content) > 200 else p.content,
                score=float(p.score),
                source_path=p.source_path,
                trust_level=TrustLevel(p.trust_level)
            ))

        return AgentResponse(
            customer_message=reason,
            actions=[
                "Faites immédiatement opposition via l’application de votre banque ou le numéro officiel au dos de la carte.",
                "Contactez votre banque via les canaux officiels (espace client / agence) pour lancer la contestation.",
                "Ne partagez jamais de CVV/PIN/OTP (même à un conseiller)."
            ],
            missing_info_questions=[],
            citations=citations,
            risk_flags=[RiskFlag(flag_type="sensitive_data_shared", description="Le client a tenté de partager des données sensibles.", severity="high")],
            raw_passages_used=len(passages),
            info_not_found=False
        )

    def _build_off_topic_response(self) -> AgentResponse:
        return AgentResponse(
            customer_message="Je peux vous aider uniquement sur la fraude bancaire (opposition, contestation, remboursement, virement/prélèvement). Quelle est votre situation exacte ?",
            actions=[],
            missing_info_questions=[
                "S’agit-il d’un paiement carte, d’un virement ou d’un prélèvement ?",
                "Est-ce une opération que vous n’avez pas autorisée ?"
            ],
            citations=[],
            risk_flags=[],
            raw_passages_used=0,
            info_not_found=True
        )

    def _build_fallback_response(self, reason: str, passages: List[RetrievedPassage]) -> AgentResponse:
        return AgentResponse(
            customer_message="Je rencontre un problème technique. Pour votre sécurité, contactez directement votre banque via les canaux officiels (application, espace client, numéro au dos de la carte).",
            actions=[
                "Contactez immédiatement votre banque via les canaux officiels",
                "Ne communiquez jamais vos codes confidentiels",
            ],
            missing_info_questions=[],
            citations=[],
            risk_flags=[RiskFlag(flag_type="technical_issue", description=reason, severity="low")],
            raw_passages_used=len(passages),
            info_not_found=True
        )

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        start_time = time.time()
        session_id = request.session_id or str(uuid.uuid4())[:8]

        agent_logger.log_request(
            session_id=session_id,
            user_message=request.user_message,
            fraud_confirmed=request.fraud_confirmed,
            transaction_context=request.transaction_context.model_dump()
        )

        sanitized_message, has_warnings, warnings = validate_user_input(request.user_message, session_id)
        if has_warnings:
            logger.warning("input_warnings", session_id=session_id, warnings=warnings)

        # fraud not confirmed
        if not request.fraud_confirmed:
            resp = AgentResponse(
                customer_message="Pouvez-vous confirmer s’il s’agit bien d’une opération non autorisée (fraude) ?",
                actions=[],
                missing_info_questions=["S’agit-il d’une transaction que vous n’avez pas effectuée ?"],
                citations=[],
                risk_flags=[],
                raw_passages_used=0,
                info_not_found=False
            )
            return ChatResponse(success=True, agent_response=resp, session_id=session_id,
                               processing_time_ms=int((time.time() - start_time) * 1000))

        # Off-topic guard
        if _looks_off_topic(sanitized_message):
            resp = self._build_off_topic_response()
            return ChatResponse(success=True, agent_response=resp, session_id=session_id,
                               processing_time_ms=int((time.time() - start_time) * 1000))

        # Sensitive data guard
        if _contains_sensitive_data(sanitized_message):
            resp = self._build_safe_refusal_response(
                session_id=session_id,
                reason="Je ne peux pas traiter ni vérifier des numéros de carte/CVV/PIN/OTP. Pour votre sécurité, faites opposition et contactez votre banque via les canaux officiels.",
                retrieval_hint="opposition carte bancaire ne communiquez jamais vos codes confidentiels contestation paiement"
            )
            return ChatResponse(success=True, agent_response=resp, session_id=session_id,
                               processing_time_ms=int((time.time() - start_time) * 1000))

        try:
            # Build retrieval query (with channel inference inside templates.py)
            tx = request.transaction_context.model_dump()
            retrieval_query = build_query_for_retrieval(sanitized_message, tx)

            retriever = get_retriever()
            passages = retriever.retrieve(query=retrieval_query, session_id=session_id)

            # If retrieval is too weak, behave as "not found"
            strong_passages = [p for p in passages if p.score >= self.MIN_PASSAGE_SCORE]
            if not strong_passages:
                resp = AgentResponse(
                    customer_message="Je ne trouve pas une procédure suffisamment précise dans la documentation actuelle pour répondre de façon fiable.",
                    actions=[],
                    missing_info_questions=[
                        "Pouvez-vous préciser s’il s’agit d’un paiement carte, d’un virement (IBAN) ou d’un prélèvement (SEPA) ?",
                        "Quel est le canal (en ligne / terminal / autre) et la date ?"
                    ],
                    citations=[],
                    risk_flags=[],
                    raw_passages_used=len(passages),
                    info_not_found=True
                )
                return ChatResponse(success=True, agent_response=resp, session_id=session_id,
                                   processing_time_ms=int((time.time() - start_time) * 1000))

            user_message = build_user_message(
                user_message=sanitized_message,
                transaction_context=tx,
                passages=strong_passages[:8],
                conversation_history=[msg.model_dump() for msg in (request.conversation_history or [])]
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]

            raw = self._call_ollama(messages)
            parsed = self._parse_json_response(raw)

            if parsed is None:
                # one retry with strict instruction
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Ta réponse n'était pas en JSON valide. Réponds UNIQUEMENT en JSON strict."})
                raw = self._call_ollama(messages)
                parsed = self._parse_json_response(raw)

            if parsed is None:
                agent_response = self._build_fallback_response("JSON parsing failed after retry", strong_passages)
            else:
                agent_response = self._validate_and_build_response(parsed, strong_passages)

            agent_logger.log_agent_decision(
                session_id=session_id,
                actions_count=len(agent_response.actions),
                citations_count=len(agent_response.citations),
                info_not_found=agent_response.info_not_found,
                risk_flags=[f.model_dump() for f in agent_response.risk_flags]
            )

            return ChatResponse(
                success=True,
                agent_response=agent_response,
                session_id=session_id,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        except Exception as e:
            logger.error("agent_error", session_id=session_id, error=str(e))
            return ChatResponse(
                success=False,
                error=str(e),
                agent_response=self._build_fallback_response(str(e), []),
                session_id=session_id,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )


_agent: Optional[FraudAssistantAgent] = None


def get_agent() -> FraudAssistantAgent:
    global _agent
    if _agent is None:
        _agent = FraudAssistantAgent()
    return _agent


async def check_ollama_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                tags = response.json()
                models = [m.get("name", "").split(":")[0] for m in tags.get("models", [])]
                return settings.ollama_model in models
    except Exception as e:
        logger.warning("ollama_health_check_failed", error=str(e))
    return False
