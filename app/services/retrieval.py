"""
Hybrid RAG retrieval module.
Combines semantic search (ChromaDB) with lexical search (BM25).
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
import numpy as np
from dataclasses import dataclass

from app.config import settings
from app.services.ingestion import DocumentChunk, ingest_all_documents
from app.services.embeddings import generate_embeddings, generate_embedding
from app.utils.validators import sanitize_rag_passage
from app.utils.logging_config import get_logger, agent_logger

logger = get_logger("retrieval")

# Collection name in ChromaDB
COLLECTION_NAME = "fraud_agent_docs"


@dataclass
class RetrievedPassage:
    """A passage retrieved from the RAG system."""
    chunk_id: str
    doc_id: str
    title: str
    content: str
    page_or_section: str
    source_path: str
    score: float
    trust_level: str = "trusted"
    retrieval_method: str = "semantic"  # or "bm25" or "hybrid"
    
    def to_citation_dict(self) -> Dict:
        """Convert to citation format for response."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "page_or_section": self.page_or_section,
            "excerpt": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "score": round(self.score, 4),
            "source_path": self.source_path,
            "trust_level": self.trust_level
        }


class HybridRetriever:
    """
    Hybrid retrieval combining:
    - Semantic search via ChromaDB (dense embeddings)
    - Lexical search via BM25 (exact term matching)
    
    This is crucial for banking procedures with specific terminology.
    """
    
    def __init__(self):
        self._chroma_client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._bm25: Optional[BM25Okapi] = None
        self._chunks_for_bm25: List[DocumentChunk] = []
        self._initialized = False
    
    @property
    def chroma_client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            persist_dir = str(settings.vectordb_dir.absolute())
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            
            self._chroma_client = chromadb.Client(ChromaSettings(
                persist_directory=persist_dir,
                anonymized_telemetry=False,
                is_persistent=True
            ))
            logger.info("chromadb_client_initialized", persist_dir=persist_dir)
        
        return self._chroma_client
    
    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the document collection."""
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("collection_ready", name=COLLECTION_NAME, count=self._collection.count())
        
        return self._collection
    
    def index_documents(self, chunks: List[DocumentChunk]) -> int:
        """
        Index document chunks in both ChromaDB (semantic) and BM25 (lexical).
        
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            logger.warning("no_chunks_to_index")
            return 0
        
        # Clear existing data
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
            logger.info("existing_collection_cleared")
        except Exception:
            pass
        
        # Reset collection reference
        self._collection = None
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "page_or_section": chunk.page_or_section,
                "source_path": chunk.source_path,
                "original_filename": chunk.metadata.get("original_filename", "")
            }
            for chunk in chunks
        ]
        
        # Generate embeddings
        logger.info("generating_embeddings", count=len(documents))
        embeddings = generate_embeddings(documents)
        
        # Index in ChromaDB
        if len(ids) != len(set(ids)):
            from collections import Counter
            duplicates = [item for item, count in Counter(ids).items() if count > 1]
            logger.error("duplicate_ids_detected", count=len(ids)-len(set(ids)), examples=duplicates[:5])
            raise ValueError(f"Expected IDs to be unique, found {len(ids)-len(set(ids))} duplicated IDs: {', '.join(duplicates[:3])}...")

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        # Build BM25 index
        self._build_bm25_index(chunks)
        
        self._initialized = True
        logger.info("indexing_complete", chunks_indexed=len(chunks))
        
        return len(chunks)
    
    def _build_bm25_index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks_for_bm25 = chunks
        
        # Tokenize documents for BM25
        tokenized_corpus = [
            self._tokenize_for_bm25(chunk.content)
            for chunk in chunks
        ]
        
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("bm25_index_built", documents=len(chunks))
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase, split on non-alphanumeric, filter short tokens
        import re
        tokens = re.findall(r'\b\w{2,}\b', text.lower())
        return tokens
    
    def _search_semantic(self, query: str, top_k: int) -> List[RetrievedPassage]:
        """Search using semantic embeddings."""
        if self.collection.count() == 0:
            return []
        
        query_embedding = generate_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        passages = []
        for i, chunk_id in enumerate(results["ids"][0]):
            # Convert distance to similarity score (cosine distance -> similarity)
            distance = results["distances"][0][i]
            score = 1 - distance  # Cosine similarity
            
            metadata = results["metadatas"][0][i]
            content = results["documents"][0][i]
            
            passages.append(RetrievedPassage(
                chunk_id=chunk_id,
                doc_id=metadata.get("doc_id", "unknown"),
                title=metadata.get("title", "Unknown"),
                content=content,
                page_or_section=metadata.get("page_or_section", "N/A"),
                source_path=metadata.get("source_path", ""),
                score=score,
                retrieval_method="semantic"
            ))
        
        return passages
    
    def _search_bm25(self, query: str, top_k: int) -> List[RetrievedPassage]:
        """Search using BM25 lexical matching."""
        if self._bm25 is None or not self._chunks_for_bm25:
            return []
        
        query_tokens = self._tokenize_for_bm25(query)
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        passages = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                chunk = self._chunks_for_bm25[idx]
                passages.append(RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    content=chunk.content,
                    page_or_section=chunk.page_or_section,
                    source_path=chunk.source_path,
                    score=float(scores[idx]),
                    retrieval_method="bm25"
                ))
        
        return passages
    
    def _load_all_chunks_from_db(self) -> List[DocumentChunk]:
        """Load all chunks from ChromaDB to rebuild BM25 index."""
        if self.collection.count() == 0:
            return []
            
        # ChromaDB get() limits to 100 results by default, we need all
        count = self.collection.count()
        results = self.collection.get(limit=count, include=["documents", "metadatas"])
        
        chunks = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            chunks.append(DocumentChunk(
                chunk_id=doc_id,
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                content=results["documents"][i],
                page_or_section=meta.get("page_or_section", ""),
                source_path=meta.get("source_path", ""),
                metadata={"original_filename": meta.get("original_filename", "")}
            ))
        return chunks

    def retrieve(self, query: str, session_id: str = "",
                 top_k_semantic: int = None, 
                 top_k_bm25: int = None) -> List[RetrievedPassage]:
        """
        Hybrid retrieval combining semantic and BM25 search.
        
        Returns:
            List of RetrievedPassage, deduplicated and scored
        """
        top_k_semantic = top_k_semantic or settings.top_k_semantic
        top_k_bm25 = top_k_bm25 or settings.top_k_bm25
        
        # Lazy load BM25 if not ready but DB has data
        if self._bm25 is None and self.get_document_count() > 0:
            logger.info("bm25_lazy_load_triggered")
            chunks = self._load_all_chunks_from_db()
            self._build_bm25_index(chunks)
        
        # Perform both searches
        semantic_results = self._search_semantic(query, top_k_semantic)
        bm25_results = self._search_bm25(query, top_k_bm25)
        
        # Merge and deduplicate
        seen_chunks = set()
        merged = []
        
        # Normalize scores for fair comparison
        semantic_weight = settings.hybrid_semantic_weight
        bm25_weight = 1 - semantic_weight
        
        # Add semantic results with weighted score
        for passage in semantic_results:
            if passage.chunk_id not in seen_chunks:
                passage.score = passage.score * semantic_weight
                passage.retrieval_method = "hybrid"
                seen_chunks.add(passage.chunk_id)
                merged.append(passage)
        
        # Add BM25 results, merging scores if duplicate
        max_bm25_score = max((p.score for p in bm25_results), default=1.0)
        for passage in bm25_results:
            normalized_bm25_score = passage.score / max_bm25_score if max_bm25_score > 0 else 0
            
            if passage.chunk_id in seen_chunks:
                # Find and update existing passage
                for existing in merged:
                    if existing.chunk_id == passage.chunk_id:
                        existing.score += normalized_bm25_score * bm25_weight
                        break
            else:
                passage.score = normalized_bm25_score * bm25_weight
                passage.retrieval_method = "hybrid"
                seen_chunks.add(passage.chunk_id)
                merged.append(passage)
        
        # Sort by combined score
        merged.sort(key=lambda x: x.score, reverse=True)
        
        # Apply security filtering and mark trust levels
        filtered_passages = []
        for passage in merged:
            sanitized_content, trust_level = sanitize_rag_passage(
                passage.content,
                passage.chunk_id,
                session_id
            )
            passage.content = sanitized_content
            passage.trust_level = trust_level
            filtered_passages.append(passage)
        
        # Log retrieval for observability
        agent_logger.log_retrieval(
            session_id=session_id,
            query=query,
            semantic_results=len(semantic_results),
            bm25_results=len(bm25_results),
            passages=[p.to_citation_dict() for p in filtered_passages[:5]]
        )
        
        return filtered_passages
    
    def get_document_count(self) -> int:
        """Get number of indexed chunks."""
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def is_initialized(self) -> bool:
        """Check if the retriever has been initialized with documents."""
        return self._initialized or self.get_document_count() > 0


# Module-level singleton
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get or create the hybrid retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def initialize_retriever(force_reindex: bool = False) -> Tuple[int, List[str]]:
    """
    Initialize the retriever.
    
    If index exists and !force_reindex:
      - Load existing count
      - Rebuild BM25 from DB (lazy loading handled in retrieve, or explicit here)
    
    If force_reindex or index empty:
      - Run full ingestion pipeline
    """
    retriever = get_retriever()
    errors = []
    
    current_count = retriever.get_document_count()
    
    if not force_reindex and current_count > 0:
        logger.info("using_existing_index", count=current_count)
        
        # If BM25 is missing (startup), we can rebuild it from DB now
        # avoiding re-reading PDF/MD files
        if retriever._bm25 is None:
            logger.info("rebuilding_bm25_from_db")
            try:
                chunks = retriever._load_all_chunks_from_db()
                retriever._build_bm25_index(chunks)
            except Exception as e:
                logger.error("bm25_rebuild_failed", error=str(e))
                # Fallback to full reindex if db read fails?
                # force_reindex = True
        
        return current_count, errors
    
    # If we need to ingest
    try:
        chunks = ingest_all_documents()
        count = retriever.index_documents(chunks)
        return count, errors
    except FileNotFoundError as e:
        errors.append(str(e))
        return 0, errors
    except Exception as e:
        errors.append(f"Indexing failed: {str(e)}")
        logger.error("indexing_failed", error=str(e))
        return 0, errors
