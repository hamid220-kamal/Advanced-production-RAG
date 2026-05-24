from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
from typing import Dict
from .schema import QueryRequest, QueryResponse, DocumentChunk, ChunkMetadata, IngestResponse
from .ingestion.chunker import SemanticChunker
from .ingestion.parser import AdvancedParser
from .retrieval.hybrid import HybridRetriever
from .ranking.reranker import CrossEncoderReRanker
from .generation.generator import LocalLLM

app = FastAPI(title="Advanced Production RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProductionRAG:
    def __init__(self):
        self.chunker = SemanticChunker(embedding_model_name="all-MiniLM-L6-v2")
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.generator = LocalLLM(model_name="google/flan-t5-small")
        self.parser = AdvancedParser()
        self.is_ready = len(self.retriever.sparse_retriever.doc_contents) > 0
        
        # Async task tracking
        self.task_status: Dict[str, str] = {}
        
        # Conversation memory tracking
        self.sessions: Dict[str, list] = {}

    def ingest_text_sync(self, text: str, source: str):
        chunks, mapping = self.chunker.chunk_document(text, source)
        self.retriever.index_documents(chunks)
        self.is_ready = True
        return len(chunks)

    def ingest_task_worker(self, task_id: str, text: str, source: str):
        self.task_status[task_id] = "processing"
        try:
            self.ingest_text_sync(text, source)
            self.task_status[task_id] = "completed"
        except Exception as e:
            self.task_status[task_id] = f"failed: {str(e)}"

    def query(self, user_query: str, top_k: int = 4, use_hyde: bool = True, session_id: str = None):
        if not self.is_ready:
            raise ValueError("System has no indexed documents. Please ingest data first.")
            
        search_query = user_query
        
        # 0. Conversation Memory: Rewrite query if session exists
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            else:
                search_query = self.generator.rewrite_query(user_query, self.sessions[session_id])
            
        # Optional: HyDE (Hypothetical Document Embeddings)
        if use_hyde:
            hypothetical_doc = self.generator.generate_hyde(search_query)
            search_query = f"{search_query} {hypothetical_doc}"
            
        # 1. Dual-Path Retrieval + RRF (get top 20 candidates using search_query)
        candidates = self.retriever.search(search_query, top_k=20)
        
        # 2. Cross-Encoder Re-ranking (get top K absolute using original query)
        final_results = self.reranker.rerank(user_query, candidates, top_k=top_k)
        
        # 3. Format the context
        context_str = self._format_prompt_context(final_results)
        
        # 4. Generate Final Answer
        generated_answer = self.generator.generate_answer(user_query, context_str)
        
        # 5. Save to session
        if session_id:
            self.sessions[session_id].append({"role": "user", "content": user_query})
            self.sessions[session_id].append({"role": "assistant", "content": generated_answer})
            
        return generated_answer, final_results
        
    def _format_prompt_context(self, chunks: list) -> str:
        return " ".join([chunk['content'] for chunk in chunks])

# Global pipeline instance
rag_pipeline = ProductionRAG()

@app.post("/api/v1/ingest/async", response_model=IngestResponse)
async def ingest_async_endpoint(payload: dict, background_tasks: BackgroundTasks):
    text = payload.get("text")
    source = payload.get("source", "api_upload")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field.")
        
    task_id = str(uuid.uuid4())
    rag_pipeline.task_status[task_id] = "queued"
    
    background_tasks.add_task(rag_pipeline.ingest_task_worker, task_id, text, source)
    
    return IngestResponse(status="queued", task_id=task_id)

@app.get("/api/v1/ingest/status/{task_id}")
async def ingest_status_endpoint(task_id: str):
    status = rag_pipeline.task_status.get(task_id, "unknown")
    return {"task_id": task_id, "status": status}

@app.post("/api/v1/ingest")
async def ingest_sync_endpoint(payload: dict):
    """Synchronous fallback for testing."""
    text = payload.get("text")
    source = payload.get("source", "api_upload")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field.")
        
    num_chunks = rag_pipeline.ingest_text_sync(text, source)
    return {"status": "success", "chunks_indexed": num_chunks}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    start_time = time.time()
    try:
        generated_answer, results = rag_pipeline.query(
            request.user_query, 
            top_k=request.top_k, 
            use_hyde=request.use_hyde,
            session_id=request.session_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    response_chunks = []
    for res in results:
        meta = ChunkMetadata(
            chunk_id=res["chunk_id"],
            parent_id=res.get("parent_id"),
            start_index=res.get("start_index", 0),
            end_index=res.get("end_index", 0),
            source=res.get("source", "unknown")
        )
        response_chunks.append(DocumentChunk(content=res["content"], metadata=meta))
        
    processing_time = (time.time() - start_time) * 1000
    
    return QueryResponse(
        query=request.user_query,
        generated_answer=generated_answer,
        results=response_chunks,
        processing_time_ms=processing_time
    )

@app.post("/api/v1/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    if not rag_pipeline.is_ready:
        raise HTTPException(status_code=400, detail="System has no indexed documents. Please ingest data first.")
        
    search_query = request.user_query
    session_id = request.session_id
    
    if session_id:
        if session_id not in rag_pipeline.sessions:
            rag_pipeline.sessions[session_id] = []
        else:
            search_query = rag_pipeline.generator.rewrite_query(request.user_query, rag_pipeline.sessions[session_id])
            
    if request.use_hyde:
        hypothetical_doc = rag_pipeline.generator.generate_hyde(search_query)
        search_query = f"{search_query} {hypothetical_doc}"
        
    candidates = rag_pipeline.retriever.search(search_query, top_k=20)
    final_results = rag_pipeline.reranker.rerank(request.user_query, candidates, top_k=request.top_k)
    context_str = rag_pipeline._format_prompt_context(final_results)
    
    def event_stream():
        import json
        # Yield metadata first
        res_list = []
        for r in final_results:
            res_list.append({
                "chunk_id": r["chunk_id"],
                "content": r["content"],
                "source": r.get("source", "unknown")
            })
        yield f"event: metadata\ndata: {json.dumps(res_list)}\n\n"
        
        # Stream answer
        generated_answer = ""
        for token in rag_pipeline.generator.stream_answer(request.user_query, context_str):
            generated_answer += token
            data = json.dumps({"token": token})
            yield f"event: token\ndata: {data}\n\n"
            
        # Save session
        if session_id:
            rag_pipeline.sessions[session_id].append({"role": "user", "content": request.user_query})
            rag_pipeline.sessions[session_id].append({"role": "assistant", "content": generated_answer})
            
        yield "event: done\ndata: {}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_stream(), media_type="text/event-stream")
