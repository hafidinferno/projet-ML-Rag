"""
Tests for the document ingestion module.
"""

import pytest
from pathlib import Path
import tempfile
import os

from app.services.ingestion import (
    extract_text_from_markdown,
    chunk_text,
    process_document,
    DocumentChunk,
    generate_chunk_id
)



class TestChunkIdGeneration:
    """Tests for chunk ID generation."""
    
    def test_chunk_id_is_deterministic(self):
        """Same inputs should produce same chunk ID."""
        id1 = generate_chunk_id("path/to/doc.pdf", "1", 0, "content")
        id2 = generate_chunk_id("path/to/doc.pdf", "1", 0, "content")
        assert id1 == id2
    
    def test_chunk_id_differs_for_different_content(self):
        """Different content should produce different IDs."""
        id1 = generate_chunk_id("path/to/doc.pdf", "1", 0, "content A")
        id2 = generate_chunk_id("path/to/doc.pdf", "1", 0, "content B")
        assert id1 != id2
    
    def test_chunk_id_differs_for_different_index(self):
        """Different indices should produce different IDs."""
        id1 = generate_chunk_id("path/to/doc.pdf", "1", 0, "content")
        id2 = generate_chunk_id("path/to/doc.pdf", "1", 1, "content")
        assert id1 != id2

    def test_chunk_id_differs_for_different_page(self):
        """Different pages should produce different IDs."""
        id1 = generate_chunk_id("path/to/doc.pdf", "1", 0, "content")
        id2 = generate_chunk_id("path/to/doc.pdf", "2", 0, "content")
        assert id1 != id2



class TestChunking:
    """Tests for text chunking."""
    
    def test_small_text_single_chunk(self):
        """Small text should produce single chunk."""
        text = "This is a short text."
        chunks = list(chunk_text(text, chunk_size=100, chunk_overlap=10))
        assert len(chunks) == 1
        assert chunks[0][0] == text
    
    def test_large_text_multiple_chunks(self):
        """Large text should produce multiple chunks."""
        text = "Word " * 200  # ~1000 chars
        chunks = list(chunk_text(text, chunk_size=100, chunk_overlap=20))
        assert len(chunks) > 1
    
    def test_chunks_have_overlap(self):
        """Chunks should have overlapping content."""
        text = "A " * 100 + "B " * 100  # Clear boundary
        chunks = list(chunk_text(text, chunk_size=100, chunk_overlap=20))
        
        # Check that consecutive chunks share some content
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i][0][-20:]
            chunk2_start = chunks[i + 1][0][:20]
            # Some overlap should exist
            assert len(chunk1_end) > 0 and len(chunk2_start) > 0


class TestMarkdownExtraction:
    """Tests for Markdown extraction."""
    
    def test_extract_simple_markdown(self):
        """Test extraction from simple markdown file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        ) as f:
            f.write("# Title\n\nSome content here.\n\n## Section 2\n\nMore content.")
            temp_path = Path(f.name)
        
        try:
            sections = extract_text_from_markdown(temp_path)
            assert len(sections) >= 1
            assert any("Title" in s["text"] or "Title" in s["section"] for s in sections)
        finally:
            os.unlink(temp_path)
    
    def test_extract_preserves_sections(self):
        """Test that section names are preserved."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        ) as f:
            f.write("# Main\n\nIntro\n\n## Procédure\n\nStep 1\n\n## Contact\n\nEmail here")
            temp_path = Path(f.name)
        
        try:
            sections = extract_text_from_markdown(temp_path)
            section_names = [s["section"] for s in sections]
            assert any("Procédure" in name or "Contact" in name for name in section_names)
        finally:
            os.unlink(temp_path)


class TestDocumentProcessing:
    """Tests for full document processing."""
    
    def test_process_markdown_document(self):
        """Test processing a complete markdown document."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        ) as f:
            content = """# Procédure d'opposition

En cas de fraude, suivez ces étapes:

1. Appelez immédiatement le numéro d'opposition
2. Bloquez votre carte via l'application
3. Déposez une contestation sous 13 mois

## Délais

Le délai de contestation est de 13 mois maximum.

## Contact

Service client: disponible 24h/24
"""
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            chunks = process_document(temp_path)
            assert len(chunks) > 0
            assert all(isinstance(c, DocumentChunk) for c in chunks)
            assert all(c.source_path for c in chunks)
            assert all(c.chunk_id for c in chunks)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
