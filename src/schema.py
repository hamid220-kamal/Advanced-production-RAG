from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChunkMetadata(BaseModel):
    chunk_id: str
    parent_id: Optional[str] = None
    start_index: int
    end_index: int
    source: str = "unknown"

class DocumentChunk(BaseModel):
    content: str
    metadata: ChunkMetadata

class QueryRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None
    top_k: int = 4
    use_hyde: bool = True

class QueryResponse(BaseModel):
    query: str
    generated_answer: str
    results: List[DocumentChunk]
    processing_time_ms: float

class IngestResponse(BaseModel):
    status: str
    task_id: str

class EvalMetrics(BaseModel):
    answer_relevancy_score: float
    context_precision_score: float
