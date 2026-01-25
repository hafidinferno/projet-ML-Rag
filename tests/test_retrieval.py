"""
Tests for the retrieval module.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.retrieval import (
    RetrievedPassage,
    HybridRetriever
)
from app.services.ingestion import DocumentChunk


class TestRetrievedPassage:
    """Tests for RetrievedPassage dataclass."""
    
    def test_to_citation_dict(self):
        """Test conversion to citation format."""
        passage = RetrievedPassage(
            chunk_id="abc123",
            doc_id="procedures",
            title="Procédures Fraude",
            content="En cas de fraude, bloquez immédiatement votre carte.",
            page_or_section="Section 1",
            source_path="/docs/procedures.pdf",
            score=0.85,
            trust_level="trusted"
        )
        
        citation = passage.to_citation_dict()
        
        assert citation["chunk_id"] == "abc123"
        assert citation["doc_id"] == "procedures"
        assert citation["score"] == 0.85
        assert citation["trust_level"] == "trusted"
        assert len(citation["excerpt"]) <= 203  # 200 + "..."
    
    def test_long_content_truncated_in_excerpt(self):
        """Test that long content is truncated."""
        long_content = "A" * 500
        passage = RetrievedPassage(
            chunk_id="abc",
            doc_id="doc",
            title="Title",
            content=long_content,
            page_or_section="1",
            source_path="/path",
            score=0.5
        )
        
        citation = passage.to_citation_dict()
        assert len(citation["excerpt"]) == 203  # 200 + "..."


class TestHybridRetriever:
    """Tests for hybrid retrieval functionality."""
    
    def test_tokenize_for_bm25(self):
        """Test BM25 tokenization."""
        retriever = HybridRetriever()
        
        tokens = retriever._tokenize_for_bm25(
            "Opposition carte bancaire en cas de fraude"
        )
        
        assert "opposition" in tokens
        assert "carte" in tokens
        assert "bancaire" in tokens
        assert "fraude" in tokens
        # Short words should be filtered
        assert "en" not in tokens
        assert "de" not in tokens
    
    def test_empty_collection_returns_empty(self):
        """Test that empty collection returns empty results."""
        retriever = HybridRetriever()
        
        # Mock empty collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        retriever._collection = mock_collection
        
        results = retriever._search_semantic("query", top_k=5)
        assert results == []
    
    def test_bm25_without_index_returns_empty(self):
        """Test BM25 search without index returns empty."""
        retriever = HybridRetriever()
        
        # No BM25 index built
        results = retriever._search_bm25("query", top_k=5)
        assert results == []


class TestRetrievalIntegration:
    """Integration tests (require mock or real data)."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks for testing."""
        return [
            DocumentChunk(
                chunk_id="chunk1",
                doc_id="opposition_carte",
                title="Opposition Carte",
                content="Pour faire opposition, appelez le numéro dédié disponible 24h/24.",
                page_or_section="Page 1",
                source_path="/docs/opposition.pdf"
            ),
            DocumentChunk(
                chunk_id="chunk2",
                doc_id="contestation",
                title="Contestation",
                content="Le délai de contestation d'une fraude est de 13 mois.",
                page_or_section="Section 2",
                source_path="/docs/contestation.md"
            ),
            DocumentChunk(
                chunk_id="chunk3",
                doc_id="securite",
                title="Sécurité Compte",
                content="Changez immédiatement vos codes d'accès en cas de suspicion.",
                page_or_section="Page 5",
                source_path="/docs/securite.pdf"
            )
        ]
    
    def test_indexing_creates_bm25(self, sample_chunks):
        """Test that indexing builds BM25 index."""
        retriever = HybridRetriever()
        
        # Mock ChromaDB to avoid actual persistence
        with patch.object(retriever, 'chroma_client'):
            mock_collection = MagicMock()
            retriever._collection = mock_collection
            
            # Just build BM25 index
            retriever._build_bm25_index(sample_chunks)
            
            assert retriever._bm25 is not None
            assert len(retriever._chunks_for_bm25) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
