"""
Agent service for the fraud assistant.
Handles LLM calls, response parsing, and validation.
"""

import json
import re
import uuid
from typing import Optional, Dict, Any, List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models.requests import ChatRequest, TransactionContext
from app.models.responses import (
    AgentResponse, ChatResponse, Citation, RiskFlag, TrustLevel
)
from app.services.retrieval import get_retriever, RetrievedPassage
from app.prompts.system_prompt import get_system_prompt
from app.prompts.templates import build_user_message, build_query_for_retrieval
from app.utils.validators import validate_user_input, is_fraud_confirmation
from app.utils.logging_config import get_logger, agent_logger

logger = get_logger("agent")


class FraudAssistantAgent:
    """
    Main agent for fraud assistance.
    Combines RAG retrieval with Mistral LLM via Ollama.
    """
    
    def __init__(self):
        self.system_prompt = get_system_prompt()
        self.http_client = httpx.Client(timeout=settings.ollama_timeout)
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'http_client'):
            self.http_client.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """
        Call Ollama API with retry logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Raw response content from the model
        """
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistency
                "top_p": 0.9,
                "num_predict": 2048
            }
        }
        
        try:
            response = self.http_client.post(
                settings.ollama_chat_url,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except httpx.HTTPStatusError as e:
            logger.error("ollama_http_error", status=e.response.status_code)
            raise
        except httpx.RequestError as e:
            logger.error("ollama_request_error", error=str(e))
            raise
    
    def _parse_json_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from the model response.
        Handles edge cases like markdown code blocks.
        """
        # Try direct JSON parse
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, raw_response)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON object directly
        brace_pattern = r'\{[\s\S]*\}'
        matches = re.findall(brace_pattern, raw_response)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        logger.warning("json_parse_failed", raw_length=len(raw_response))
        return None
    
    def _validate_and_build_response(
        self,
        parsed: Dict[str, Any],
        passages: List[RetrievedPassage]
    ) -> AgentResponse:
        """
        Validate parsed JSON and build AgentResponse.
        Fills in missing fields with defaults.
        """
        # Build citations from parsed data + passage metadata
        citations = []
        parsed_citations = parsed.get("citations", [])
        
        # Map passages for enrichment
        passage_map = {p.doc_id: p for p in passages}
        
        for i, cit in enumerate(parsed_citations):
            doc_id = cit.get("doc_id", f"unknown_{i}")
            passage = passage_map.get(doc_id)
            
            citations.append(Citation(
                chunk_id=passage.chunk_id if passage else f"parsed_{i}",
                doc_id=doc_id,
                title=cit.get("title", passage.title if passage else doc_id),
                page_or_section=cit.get("page_or_section", "N/A"),
                excerpt=cit.get("excerpt", "")[:200],
                score=passage.score if passage else 0.0,
                source_path=passage.source_path if passage else "",
                trust_level=TrustLevel(passage.trust_level) if passage else TrustLevel.TRUSTED
            ))
        
        # Build risk flags
        risk_flags = []
        for flag in parsed.get("risk_flags", []):
            if isinstance(flag, dict):
                risk_flags.append(RiskFlag(
                    flag_type=flag.get("flag_type", "unknown"),
                    description=flag.get("description", ""),
                    severity=flag.get("severity", "medium")
                ))
        
        return AgentResponse(
            customer_message=parsed.get("customer_message", 
                "Je n'ai pas pu traiter votre demande. Veuillez contacter votre banque directement."),
            actions=parsed.get("actions", []),
            missing_info_questions=parsed.get("missing_info_questions", []),
            citations=citations,
            risk_flags=risk_flags,
            raw_passages_used=len(passages),
            info_not_found=parsed.get("info_not_found", False)
        )
    
    def _build_fallback_response(
        self,
        reason: str,
        passages: List[RetrievedPassage]
    ) -> AgentResponse:
        """Build a safe fallback response when LLM fails."""
        return AgentResponse(
            customer_message=(
                "Je rencontre actuellement des difficultés techniques. "
                "Pour votre sécurité, je vous recommande de contacter directement "
                "votre banque via les numéros officiels figurant sur votre carte bancaire "
                "ou sur votre espace client."
            ),
            actions=[
                "Appelez le numéro au dos de votre carte bancaire",
                "Connectez-vous à votre espace bancaire en ligne",
                "Ne communiquez jamais vos codes confidentiels"
            ],
            missing_info_questions=[],
            citations=[],
            risk_flags=[
                RiskFlag(
                    flag_type="technical_issue",
                    description=reason,
                    severity="low"
                )
            ],
            raw_passages_used=len(passages),
            info_not_found=True
        )
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request from a client.
        
        This is the main entry point for the fraud assistant.
        """
        import time
        start_time = time.time()
        
        session_id = request.session_id or str(uuid.uuid4())[:8]
        
        # Log the request
        agent_logger.log_request(
            session_id=session_id,
            user_message=request.user_message,
            fraud_confirmed=request.fraud_confirmed,
            transaction_context=request.transaction_context.model_dump()
        )
        
        # Validate input
        sanitized_message, has_warnings, warnings = validate_user_input(
            request.user_message,
            session_id
        )
        
        if has_warnings:
            logger.warning("input_warnings", session_id=session_id, warnings=warnings)
        
        # Check fraud confirmation
        if not request.fraud_confirmed:
            # If fraud not confirmed, return early
            return ChatResponse(
                success=True,
                agent_response=AgentResponse(
                    customer_message=(
                        "Je suis là pour vous aider en cas de fraude confirmée. "
                        "Pouvez-vous confirmer qu'il s'agit bien d'une transaction frauduleuse?"
                    ),
                    actions=[],
                    missing_info_questions=[
                        "S'agit-il bien d'une transaction que vous n'avez pas effectuée?"
                    ],
                    citations=[],
                    risk_flags=[],
                    raw_passages_used=0,
                    info_not_found=False
                ),
                session_id=session_id,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        try:
            # Build retrieval query
            retrieval_query = build_query_for_retrieval(
                sanitized_message,
                request.transaction_context.model_dump()
            )
            
            # Retrieve relevant passages
            retriever = get_retriever()
            passages = retriever.retrieve(
                query=retrieval_query,
                session_id=session_id
            )
            
            # Build messages for LLM
            user_message = build_user_message(
                user_message=sanitized_message,
                transaction_context=request.transaction_context.model_dump(),
                passages=passages,
                conversation_history=[
                    msg.model_dump() for msg in (request.conversation_history or [])
                ]
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Call LLM
            raw_response = self._call_ollama(messages)
            
            # Parse response
            parsed = self._parse_json_response(raw_response)
            
            if parsed is None:
                # Retry once with explicit JSON instruction
                messages.append({
                    "role": "assistant",
                    "content": raw_response
                })
                messages.append({
                    "role": "user",
                    "content": "Ta réponse n'était pas en JSON valide. Reformule en JSON strict."
                })
                
                raw_response = self._call_ollama(messages)
                parsed = self._parse_json_response(raw_response)
            
            if parsed is None:
                agent_response = self._build_fallback_response(
                    "JSON parsing failed after retry",
                    passages
                )
            else:
                agent_response = self._validate_and_build_response(parsed, passages)
            
            # Log the decision
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


# Module-level singleton
_agent: Optional[FraudAssistantAgent] = None


def get_agent() -> FraudAssistantAgent:
    """Get or create the agent singleton."""
    global _agent
    if _agent is None:
        _agent = FraudAssistantAgent()
    return _agent


async def check_ollama_health() -> bool:
    """Check if Ollama is available and the model is loaded."""
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
