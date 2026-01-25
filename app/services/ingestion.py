"""
Document ingestion module for PDF and Markdown files.
Handles text extraction, chunking, and metadata preservation.
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field
import fitz  # pymupdf
import markdown
from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger("ingestion")


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    chunk_id: str
    doc_id: str
    title: str
    content: str
    page_or_section: str
    source_path: str
    start_char: int = 0
    end_char: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "page_or_section": self.page_or_section,
            "source_path": self.source_path,
            "start_char": self.start_char,
            "end_char": self.end_char,
            **self.metadata
        }

def generate_chunk_id(source_path: str, page_or_section: str, chunk_index: int, content: str) -> str:
    """
    Generate a unique, stable chunk ID using SHA1.
    
    Uniqueness is guaranteed by combining:
    - source_path: full path to the document
    - page_or_section: page number or section name
    - chunk_index: index within the document
    - content: hash of the full chunk content
    """
    import hashlib
    # Hash the content separately for stability
    content_hash = hashlib.sha1(content.encode('utf-8')).hexdigest()[:12]
    # Combine all uniqueness factors
    unique_input = f"{source_path}|{page_or_section}|{chunk_index}|{content_hash}"
    return hashlib.sha1(unique_input.encode('utf-8')).hexdigest()[:24]


def extract_text_from_pdf(file_path: Path) -> List[Dict]:
    """
    Extract text from PDF with page-level metadata.
    
    Returns:
        List of dicts with 'text', 'page', 'title' keys
    """
    pages = []
    try:
        doc = fitz.open(str(file_path))
        title = file_path.stem.replace("_", " ").replace("-", " ")
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "text": text,
                    "page": page_num,
                    "title": title,
                    "total_pages": len(doc)
                })
        
        doc.close()
        logger.info("pdf_extracted", file=str(file_path), pages=len(pages))
        
    except Exception as e:
        logger.error("pdf_extraction_failed", file=str(file_path), error=str(e))
        raise
    
    return pages


def extract_text_from_markdown(file_path: Path) -> List[Dict]:
    """
    Extract text from Markdown with section-level metadata.
    Preserves heading structure for better chunking.
    
    Returns:
        List of dicts with 'text', 'section', 'title' keys
    """
    sections = []
    try:
        content = file_path.read_text(encoding="utf-8")
        title = file_path.stem.replace("_", " ").replace("-", " ")
        
        # Split by headings (##, ###, etc.) while preserving heading text
        heading_pattern = r'^(#{1,4})\s+(.+)$'
        parts = re.split(heading_pattern, content, flags=re.MULTILINE)
        
        current_section = "Introduction"
        current_text = ""
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            if i + 2 < len(parts) and parts[i].startswith("#"):
                # This is a heading marker
                if current_text.strip():
                    sections.append({
                        "text": current_text.strip(),
                        "section": current_section,
                        "title": title
                    })
                current_section = parts[i + 1].strip()
                current_text = f"{parts[i]} {parts[i + 1]}\n"
                i += 2
            else:
                current_text += part + "\n"
                i += 1
        
        # Don't forget the last section
        if current_text.strip():
            sections.append({
                "text": current_text.strip(),
                "section": current_section,
                "title": title
            })
        
        # If no sections found, treat entire document as one section
        if not sections:
            sections.append({
                "text": content,
                "section": "Document complet",
                "title": title
            })
        
        logger.info("markdown_extracted", file=str(file_path), sections=len(sections))
        
    except Exception as e:
        logger.error("markdown_extraction_failed", file=str(file_path), error=str(e))
        raise
    
    return sections


def split_into_chunks(text: str, chunk_size: int = None, 
               chunk_overlap: int = None) -> Generator[tuple, None, None]:
    """
    Split text into overlapping chunks.
    Tries to break at sentence boundaries when possible.
    
    Yields:
        Tuples of (chunk_content, start_char, end_char)
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    # Clean text
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    if len(text) <= chunk_size:
        yield (text, 0, len(text))
        return
    
    # Sentence boundary patterns for better chunking
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-Ÿ])')
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            yield (text[start:], start, len(text))
            break
        
        # Try to find a sentence boundary near the end
        search_start = max(start + chunk_size - 100, start)
        search_text = text[search_start:end]
        
        # Look for sentence ending in the last 100 chars
        matches = list(sentence_endings.finditer(search_text))
        if matches:
            # Use the last sentence boundary
            boundary = search_start + matches[-1].end()
            if boundary > start:
                end = boundary
        else:
            # Fall back to space
            space_pos = text.rfind(' ', start + chunk_size - 50, end)
            if space_pos > start:
                end = space_pos
        
        yield (text[start:end].strip(), start, end)
        start = end - chunk_overlap


def process_document(file_path: Path) -> List[DocumentChunk]:
    """
    Process a single document (PDF or MD) into chunks.
    
    Returns:
        List of DocumentChunk objects
    """
    doc_id = file_path.stem
    source_path = str(file_path.absolute())
    chunks = []
    
    # Extract based on file type
    if file_path.suffix.lower() == ".pdf":
        sections = extract_text_from_pdf(file_path)
        page_key = "page"
    elif file_path.suffix.lower() in [".md", ".markdown"]:
        sections = extract_text_from_markdown(file_path)
        page_key = "section"
    else:
        logger.warning("unsupported_file_type", file=str(file_path))
        return []
    
    # Chunk each section
    chunk_index = 0
    for section in sections:
        text = section["text"]
        title = section["title"]
        page_or_section = str(section.get(page_key, "N/A"))
        
        for chunk_content, start_char, end_char in split_into_chunks(text):
            if len(chunk_content.strip()) < 20:
                continue  # Skip tiny chunks
            
            chunk_id = generate_chunk_id(source_path, page_or_section, chunk_index, chunk_content)
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                content=chunk_content,
                page_or_section=page_or_section,
                source_path=source_path,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "file_type": file_path.suffix.lower(),
                    "original_filename": file_path.name
                }
            ))
            chunk_index += 1
    
    logger.info(
        "document_processed",
        doc_id=doc_id,
        chunks_created=len(chunks)
    )
    
    return chunks


def ingest_all_documents(docs_dir: Path = None) -> List[DocumentChunk]:
    """
    Process all documents in the specified directory.
    
    Args:
        docs_dir: Directory containing PDF and MD files
    
    Returns:
        List of all DocumentChunk objects
    """
    docs_dir = docs_dir or settings.docs_dir
    
    if not docs_dir.exists():
        logger.error("docs_dir_not_found", path=str(docs_dir))
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    all_chunks = []
    processed_files = 0
    errors = []
    
    # Find all PDF and MD files
    patterns = ["*.pdf", "*.PDF", "*.md", "*.MD", "*.markdown"]
    files = set()
    for pattern in patterns:
        for file in docs_dir.glob(pattern):
            files.add(file.resolve())
    
    logger.info("ingestion_started", files_found=len(files), docs_dir=str(docs_dir))
    
    for file_path in sorted(list(files)):
        try:
            chunks = process_document(file_path)
            all_chunks.extend(chunks)
            processed_files += 1
        except Exception as e:
            error_msg = f"{file_path.name}: {str(e)}"
            errors.append(error_msg)
            logger.error("document_processing_failed", file=str(file_path), error=str(e))
    
    logger.info(
        "ingestion_complete",
        files_processed=processed_files,
        total_chunks=len(all_chunks),
        errors=len(errors)
    )
    
    return all_chunks
