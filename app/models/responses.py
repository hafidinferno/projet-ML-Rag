"""
Pydantic response models for the Fraud Agent API.
Includes auditable citation structure with chunk_id, score, source_path.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TrustLevel(str, Enum):
    """Trust level for RAG passages."""
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"  # Flagged by injection filter


class Citation(BaseModel):
    """
    Auditable citation structure.
    Each piece of information from the agent must be traceable.
    """
    chunk_id: str = Field(..., description="Unique identifier of the chunk in vector store")
    doc_id: str = Field(..., description="Document identifier (filename)")
    title: str = Field(..., description="Document title or filename")
    page_or_section: str = Field(..., description="Page number or section name")
    excerpt: str = Field(..., description="Relevant text excerpt (max 200 chars)")
    score: float = Field(..., description="Retrieval score (higher = more relevant)")
    source_path: str = Field(..., description="Full path to source document")
    trust_level: TrustLevel = Field(default=TrustLevel.TRUSTED)


class RiskFlag(BaseModel):
    """Risk indicator for the account."""
    flag_type: str = Field(..., description="Type: account_compromised, multiple_fraud, urgent_action")
    description: str
    severity: str = Field(..., description="low, medium, high, critical")


class AgentResponse(BaseModel):
    """
    Structured response from the fraud agent.
    Designed for n8n consumption + human-readable message.
    """
    customer_message: str = Field(
        ..., 
        description="Human-readable message for the customer in their language"
    )
    actions: List[str] = Field(
        default_factory=list,
        description="List of recommended action steps for the customer"
    )
    missing_info_questions: List[str] = Field(
        default_factory=list,
        description="Questions to ask if information is missing"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Auditable sources for each piece of information"
    )
    risk_flags: List[RiskFlag] = Field(
        default_factory=list,
        description="Risk indicators if account may be compromised"
    )
    raw_passages_used: int = Field(
        default=0,
        description="Number of RAG passages injected in prompt"
    )
    info_not_found: bool = Field(
        default=False,
        description="True if required info was not found in documents"
    )
    
    
class ChatResponse(BaseModel):
    """Full API response for /chat endpoint."""
    success: bool
    agent_response: Optional[AgentResponse] = None
    error: Optional[str] = None
    session_id: Optional[str] = None
    processing_time_ms: int = Field(default=0)
    

class IngestResponse(BaseModel):
    """Response for document ingestion."""
    success: bool
    documents_processed: int = 0
    chunks_created: int = 0
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_available: bool
    vectordb_ready: bool
    documents_indexed: int
    version: str = "1.0.0"
