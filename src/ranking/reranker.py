from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class CrossEncoderReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes a lightweight local cross-encoder model.
        Used ms-marco-MiniLM-L-6-v2 (under 100MB) to conserve local space.
        """
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Scores each (query, candidate) pair and returns the top_k absolute highest-scoring contexts.
        """
        if not candidates:
            return []
            
        # Prepare pairs for cross-encoder
        pairs = [[query, doc["content"]] for doc in candidates]
        
        # Predict hard relevance scores
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        for idx, doc in enumerate(candidates):
            doc["cross_encoder_score"] = float(scores[idx])
            
        ranked_candidates = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)
        
        return ranked_candidates[:top_k]
