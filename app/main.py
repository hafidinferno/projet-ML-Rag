"""
FastAPI application for the Fraud Agent API.
Provides endpoints for ingestion, chat, health, and logs.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.requests import ChatRequest, IngestRequest
from app.models.responses import (
    ChatResponse, IngestResponse, HealthResponse
)
from app.services.retrieval import get_retriever, initialize_retriever
from app.services.agent import get_agent, check_ollama_health
from app.utils.logging_config import setup_logging, get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes logging only. Ingestion is triggered via /ingest endpoint.
    """
    # Setup logging
    setup_logging()
    logger.info("application_starting", version="1.0.0")
    
    # Note: We do NOT auto-ingest at startup to avoid:
    # 1. Double ingestion with uvicorn --reload
    # 2. Slow startup times
    # 3. Unexpected re-indexing
    # Use POST /ingest to initialize or refresh the index.
    
    yield
    
    # Cleanup
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title="Agent Fraude Bancaire",
    description="API pour l'agent contextuel d'assistance fraude bancaire avec RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
#                              ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns status of Ollama and vector database.
    """
    retriever = get_retriever()
    ollama_ok = await check_ollama_health()
    vectordb_ready = retriever.get_document_count() > 0
    
    status = "healthy" if (ollama_ok and vectordb_ready) else "degraded"
    if not ollama_ok and not vectordb_ready:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        ollama_available=ollama_ok,
        vectordb_ready=vectordb_ready,
        documents_indexed=retriever.get_document_count()
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(request: IngestRequest = None):
    """
    Ingest/reindex all documents from the docs directory.
    
    This endpoint:
    1. Scans the docs directory for PDF and MD files
    2. Extracts text and metadata
    3. Chunks the content
    4. Generates embeddings
    5. Indexes in ChromaDB + builds BM25 index
    """
    request = request or IngestRequest()
    start_time = time.time()
    
    logger.info("ingestion_requested", force_reindex=request.force_reindex)
    
    try:
        # Check if docs directory exists
        if not settings.docs_dir.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Documents directory not found: {settings.docs_dir}"
            )
        
        # Perform ingestion
        count, errors = initialize_retriever(force_reindex=request.force_reindex)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return IngestResponse(
            success=len(errors) == 0,
            documents_processed=count,
            chunks_created=count,  # Each chunk is counted
            errors=errors,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("ingestion_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint for the fraud assistance agent.
    
    This endpoint:
    1. Validates the input (injection protection)
    2. Checks if fraud is confirmed
    3. Retrieves relevant documents (hybrid: semantic + BM25)
    4. Calls Mistral via Ollama with strict prompting
    5. Parses and validates the JSON response
    6. Returns structured response with citations
    
    The agent will ONLY provide information from the indexed documents.
    """
    logger.info("chat_request", 
                fraud_confirmed=request.fraud_confirmed,
                channel=request.transaction_context.channel)
    
    # Check if retriever is initialized
    retriever = get_retriever()
    if retriever.get_document_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Document index not initialized. Please call /ingest first."
        )
    
    # Process the request
    agent = get_agent()
    response = await agent.process_chat(request)
    
    return response


@app.get("/logs", tags=["System"])
async def get_recent_logs(
    lines: int = Query(default=100, ge=1, le=1000, description="Number of lines to return")
):
    """
    Get recent application logs.
    Returns the last N lines from today's log file.
    """
    today = datetime.now().strftime("%Y%m%d")
    log_file = settings.logs_dir / f"fraud_agent_{today}.log"
    
    if not log_file.exists():
        return {"logs": [], "message": "No logs for today"}
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:]
        
        return {
            "logs": recent_lines,
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """
    List all indexed documents.
    """
    docs_path = settings.docs_dir
    
    if not docs_path.exists():
        return {"documents": [], "error": "Documents directory not found"}
    
    documents = []
    for pattern in ["*.pdf", "*.PDF", "*.md", "*.MD"]:
        for file in docs_path.glob(pattern):
            documents.append({
                "filename": file.name,
                "type": file.suffix.lower(),
                "size_bytes": file.stat().st_size,
                "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            })
    
    retriever = get_retriever()
    
    return {
        "documents": documents,
        "total_files": len(documents),
        "chunks_indexed": retriever.get_document_count()
    }


# ============================================================================
#                              ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
