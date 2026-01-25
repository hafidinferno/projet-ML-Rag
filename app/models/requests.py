"""
Pydantic request models for the Fraud Agent API.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import re


class TransactionContext(BaseModel):
    """Context of the potentially fraudulent transaction."""
    
    amount: float = Field(..., description="Transaction amount", ge=0)
    currency: str = Field(..., description="Currency code (EUR, USD, etc.)", min_length=3, max_length=3)
    merchant: str = Field(..., description="Merchant name")
    channel: str = Field(..., description="Payment channel: online, terminal, virement, prelevement")
    date: str = Field(..., description="Transaction date (YYYY-MM-DD)")
    country: Optional[str] = Field(None, description="Country code if available")
    last_four_digits: Optional[str] = Field(
        None, 
        description="Last 4 digits of card (ONLY if necessary)",
        min_length=4,
        max_length=4
    )
    
    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        """Normalize and validate channel."""
        v = v.lower().strip()
        valid_channels = {"online", "terminal", "virement", "prelevement", "cheque", "autre"}
        if v not in valid_channels:
            v = "autre"
        return v
    
    @field_validator("last_four_digits")
    @classmethod
    def validate_last_four(cls, v: Optional[str]) -> Optional[str]:
        """Ensure only digits."""
        if v is not None and not v.isdigit():
            raise ValueError("last_four_digits must contain only digits")
        return v


class ConversationMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., description="Role: user or assistant")
    content: str


class ChatRequest(BaseModel):
    """Request payload for the /chat endpoint."""
    
    user_message: str = Field(..., description="User's message", min_length=1, max_length=2000)
    transaction_context: TransactionContext
    fraud_confirmed: bool = Field(..., description="Whether fraud is confirmed by user")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        default=None,
        description="Previous messages in this conversation"
    )
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    
    @field_validator("user_message")
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Basic sanitization - detailed injection detection is in validators.py."""
        # Remove null bytes and control characters
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)
        return v.strip()


class IngestRequest(BaseModel):
    """Request for document ingestion."""
    force_reindex: bool = Field(
        default=False,
        description="Force full reindexing even if documents haven't changed"
    )
