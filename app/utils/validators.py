"""
Security validators for anti-prompt injection and input sanitization.

Key principles:
- RAG passages are citations, not executable instructions
- Any instruction in docs/user contradicting system prompt is ignored
- Suspicious patterns are filtered and marked as untrusted
"""

import re
from typing import Tuple, List
from dataclasses import dataclass
from app.utils.logging_config import agent_logger


@dataclass
class InjectionCheckResult:
    """Result of injection check."""
    is_suspicious: bool
    patterns_found: List[str]
    sanitized_text: str
    trust_level: str  # "trusted" or "untrusted"


# Patterns that indicate potential prompt injection attempts
INJECTION_PATTERNS = [
    # Direct instruction attempts
    (r'ignore\s+(previous|above|all|the)\s+(instructions?|prompts?|rules?)', 'ignore_instructions'),
    (r'forget\s+(everything|all|what)', 'forget_command'),
    (r'disregard\s+(previous|above|the|all)', 'disregard_command'),
    
    # System prompt manipulation
    (r'(new\s+)?system\s*(prompt|instruction|message)', 'system_override'),
    (r'you\s+are\s+(now|a)\s+', 'role_override'),
    (r'act\s+as\s+(if\s+you\s+are|a)', 'role_override'),
    (r'pretend\s+(to\s+be|you\s+are)', 'role_override'),
    
    # Output manipulation
    (r'(output|print|say|respond\s+with)\s*[:\-]?\s*["\']', 'output_injection'),
    (r'your\s+(new\s+)?response\s+(should|must|will)\s+be', 'output_override'),
    
    # Jailbreak patterns
    (r'(DAN|STAN|DUDE)\s*(mode)?', 'jailbreak_pattern'),
    (r'developer\s+mode', 'jailbreak_pattern'),
    (r'(no|without)\s+(restrictions?|limits?|rules?)', 'jailbreak_pattern'),
    
    # Code execution attempts
    (r'```\s*(python|bash|shell|cmd|exec)', 'code_injection'),
    (r'(execute|run|eval)\s*\(', 'code_injection'),
    
    # Delimiter attacks
    (r'<\|?(end|start|system|im_end|im_start)\|?>', 'delimiter_attack'),
    (r'\[INST\]|\[/INST\]', 'delimiter_attack'),
    (r'###\s*(system|user|assistant)', 'delimiter_attack'),
    
    # Data exfiltration attempts
    (r'(show|reveal|display)\s+(the\s+)?(system|full|original)\s+(prompt|instructions?)', 'exfiltration'),
    (r'what\s+(are|is)\s+your\s+(instructions?|system\s+prompt)', 'exfiltration'),
]

# Sensitive data patterns that should never be requested
SENSITIVE_DATA_PATTERNS = [
    (r'(card|carte)\s*(number|numero|n°|num)', 'card_number_request'),
    (r'(cvv|cvc|cvv2|cvc2|code\s+sécurité)', 'cvv_request'),
    (r'(pin|code\s+secret|code\s+confidentiel)', 'pin_request'),
    (r'(password|mot\s+de\s+passe|mdp)', 'password_request'),
    (r'(full|complet|entier)\s*(pan|numéro)', 'full_pan_request'),
]


def check_for_injection(text: str, source: str = "unknown", 
                        session_id: str = "") -> InjectionCheckResult:
    """
    Check text for potential prompt injection patterns.
    
    Args:
        text: The text to check (user message or RAG passage)
        source: Source of the text for logging ("user" or "rag")
        session_id: Session ID for logging
    
    Returns:
        InjectionCheckResult with detection status and sanitized text
    """
    if not text:
        return InjectionCheckResult(
            is_suspicious=False,
            patterns_found=[],
            sanitized_text=text,
            trust_level="trusted"
        )
    
    patterns_found = []
    text_lower = text.lower()
    
    # Check for injection patterns
    for pattern, pattern_name in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            patterns_found.append(pattern_name)
    
    is_suspicious = len(patterns_found) > 0
    
    if is_suspicious:
        # Log the detection
        agent_logger.log_injection_detected(
            session_id=session_id,
            source=source,
            pattern=", ".join(patterns_found),
            content_preview=text[:200]
        )
    
    # Sanitize: remove null bytes, excessive whitespace, control chars
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    return InjectionCheckResult(
        is_suspicious=is_suspicious,
        patterns_found=patterns_found,
        sanitized_text=sanitized,
        trust_level="untrusted" if is_suspicious else "trusted"
    )


def check_for_sensitive_data_request(text: str) -> Tuple[bool, List[str]]:
    """
    Check if text is trying to request sensitive data.
    
    Returns:
        Tuple of (is_requesting_sensitive, list of patterns found)
    """
    patterns_found = []
    text_lower = text.lower()
    
    for pattern, pattern_name in SENSITIVE_DATA_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            patterns_found.append(pattern_name)
    
    return (len(patterns_found) > 0, patterns_found)


def sanitize_rag_passage(passage: str, chunk_id: str, 
                         session_id: str = "") -> Tuple[str, str]:
    """
    Sanitize a RAG passage and determine trust level.
    
    Passages are treated as DATA/CITATIONS only, never as instructions.
    If suspicious patterns are found, the passage is marked untrusted.
    
    Returns:
        Tuple of (sanitized_passage, trust_level)
    """
    check_result = check_for_injection(
        text=passage, 
        source=f"rag:{chunk_id}",
        session_id=session_id
    )
    
    return check_result.sanitized_text, check_result.trust_level


def validate_user_input(user_message: str, session_id: str = "") -> Tuple[str, bool, List[str]]:
    """
    Validate and sanitize user input.
    
    Returns:
        Tuple of (sanitized_message, has_warnings, warning_types)
    """
    # Check for injection
    injection_check = check_for_injection(
        text=user_message,
        source="user",
        session_id=session_id
    )
    
    # Check for sensitive data requests (user trying to make agent request sensitive data)
    is_sensitive, sensitive_patterns = check_for_sensitive_data_request(user_message)
    
    all_warnings = injection_check.patterns_found + sensitive_patterns
    
    return (
        injection_check.sanitized_text,
        len(all_warnings) > 0,
        all_warnings
    )


def is_fraud_confirmation(message: str) -> bool:
    """
    Check if the user message is confirming fraud.
    
    Matches variations of "yes, it's fraud" in French and English.
    """
    message_lower = message.lower().strip()
    
    confirmation_patterns = [
        # French
        r"oui.*fraude",
        r"oui.*c'est (une |la )?fraude",
        r"je\s+confirme.*fraude",
        r"(c'est|c est)\s+(bien\s+)?(une\s+)?fraude",
        r"fraude\s+confirm[ée]e?",
        r"effectivement.*fraude",
        
        # English
        r"yes.*fraud",
        r"it('?s| is)\s+(a\s+)?fraud",
        r"confirm.*fraud",
        r"fraud\s+confirmed",
    ]
    
    for pattern in confirmation_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False
