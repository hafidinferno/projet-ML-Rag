"""
Structured logging configuration with observability focus.
Logs: requests, RAG passages retrieved, agent decisions.
"""

import logging
import sys
import structlog
from pathlib import Path
from datetime import datetime
from app.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Ensure logs directory exists
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file handler with daily rotation pattern
    log_file = settings.logs_dir / f"fraud_agent_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Also configure standard logging for libraries
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, settings.log_level),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )


def get_logger(name: str = "fraud_agent"):
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Specialized loggers for different concerns
class AgentLogger:
    """Logger specifically for agent operations with observability focus."""
    
    def __init__(self):
        self.logger = get_logger("agent")
    
    def log_request(self, session_id: str, user_message: str, fraud_confirmed: bool, 
                    transaction_context: dict) -> None:
        """Log incoming chat request."""
        self.logger.info(
            "chat_request_received",
            session_id=session_id,
            fraud_confirmed=fraud_confirmed,
            message_length=len(user_message),
            channel=transaction_context.get("channel"),
            amount=transaction_context.get("amount")
        )
    
    def log_retrieval(self, session_id: str, query: str, 
                      semantic_results: int, bm25_results: int,
                      passages: list) -> None:
        """Log RAG retrieval results for observability."""
        self.logger.info(
            "rag_retrieval_complete",
            session_id=session_id,
            query_length=len(query),
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            top_passages=[
                {
                    "chunk_id": p.get("chunk_id"),
                    "doc_id": p.get("doc_id"),
                    "score": p.get("score"),
                    "trust_level": p.get("trust_level", "trusted")
                }
                for p in passages[:5]
            ]
        )
    
    def log_agent_decision(self, session_id: str, actions_count: int,
                           citations_count: int, info_not_found: bool,
                           risk_flags: list) -> None:
        """Log agent's decision for audit trail."""
        self.logger.info(
            "agent_decision",
            session_id=session_id,
            actions_count=actions_count,
            citations_count=citations_count,
            info_not_found=info_not_found,
            risk_flags=[f["flag_type"] for f in risk_flags] if risk_flags else []
        )
    
    def log_injection_detected(self, session_id: str, source: str, 
                                pattern: str, content_preview: str) -> None:
        """Log potential prompt injection attempts."""
        self.logger.warning(
            "injection_detected",
            session_id=session_id,
            source=source,
            pattern=pattern,
            content_preview=content_preview[:100]
        )


# Singleton instance
agent_logger = AgentLogger()
