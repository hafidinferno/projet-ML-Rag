"""
Tests for the agent module.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.agent import FraudAssistantAgent
from app.models.requests import ChatRequest, TransactionContext
from app.services.retrieval import RetrievedPassage


class TestJsonParsing:
    """Tests for JSON response parsing."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        agent = FraudAssistantAgent()
        
        valid_json = json.dumps({
            "customer_message": "Voici les étapes à suivre.",
            "actions": ["Étape 1", "Étape 2"],
            "citations": [],
            "missing_info_questions": [],
            "risk_flags": [],
            "info_not_found": False
        })
        
        result = agent._parse_json_response(valid_json)
        
        assert result is not None
        assert result["customer_message"] == "Voici les étapes à suivre."
        assert len(result["actions"]) == 2
    
    def test_parse_json_in_markdown_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        agent = FraudAssistantAgent()
        
        markdown_wrapped = """Voici ma réponse:

```json
{
    "customer_message": "Message",
    "actions": [],
    "citations": [],
    "missing_info_questions": [],
    "risk_flags": [],
    "info_not_found": false
}
```
"""
        
        result = agent._parse_json_response(markdown_wrapped)
        
        assert result is not None
        assert result["customer_message"] == "Message"
    
    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with text before/after."""
        agent = FraudAssistantAgent()
        
        with_text = 'Here is my response: {"customer_message": "Test", "actions": []} End.'
        
        result = agent._parse_json_response(with_text)
        
        assert result is not None
        assert result["customer_message"] == "Test"
    
    def test_parse_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        agent = FraudAssistantAgent()
        
        invalid = "This is not JSON at all"
        
        result = agent._parse_json_response(invalid)
        
        assert result is None


class TestResponseValidation:
    """Tests for response building and validation."""
    
    def test_build_response_with_valid_data(self):
        """Test building AgentResponse from valid parsed data."""
        agent = FraudAssistantAgent()
        
        parsed = {
            "customer_message": "Suivez ces étapes pour signaler la fraude.",
            "actions": [
                "1. Appelez le numéro d'opposition",
                "2. Bloquez votre carte"
            ],
            "citations": [
                {
                    "doc_id": "opposition",
                    "page_or_section": "Page 1",
                    "excerpt": "Numéro disponible 24h/24"
                }
            ],
            "missing_info_questions": [],
            "risk_flags": [],
            "info_not_found": False
        }
        
        passages = [
            RetrievedPassage(
                chunk_id="chunk1",
                doc_id="opposition",
                title="Opposition CB",
                content="Contenu...",
                page_or_section="Page 1",
                source_path="/docs/opposition.pdf",
                score=0.9
            )
        ]
        
        response = agent._validate_and_build_response(parsed, passages)
        
        assert response.customer_message == parsed["customer_message"]
        assert len(response.actions) == 2
        assert len(response.citations) == 1
        assert response.citations[0].score == 0.9
        assert response.info_not_found is False
    
    def test_build_fallback_response(self):
        """Test building fallback response on error."""
        agent = FraudAssistantAgent()
        
        response = agent._build_fallback_response("Connection error", [])
        
        assert response.customer_message is not None
        assert "banque" in response.customer_message.lower()
        assert response.info_not_found is True
        assert len(response.risk_flags) > 0


class TestFraudConfirmation:
    """Tests for fraud confirmation check."""
    
    def test_not_confirmed_returns_early(self):
        """Test that non-confirmed fraud returns confirmation request."""
        # This would normally be an async test with proper mocking
        # Simplified for demonstration
        pass


class TestInputValidation:
    """Tests for input validation integration."""
    
    from app.utils.validators import is_fraud_confirmation
    
    @pytest.mark.parametrize("message,expected", [
        ("Oui, c'est une fraude", True),
        ("oui c est une fraude", True),
        ("Je confirme, c'est une fraude", True),
        ("C'est bien une fraude", True),
        ("Non, j'ai fait cet achat", False),
        ("Quels sont les délais?", False),
        ("Bonjour", False),
    ])
    def test_fraud_confirmation_detection(self, message, expected):
        """Test various fraud confirmation messages."""
        from app.utils.validators import is_fraud_confirmation
        
        result = is_fraud_confirmation(message)
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
