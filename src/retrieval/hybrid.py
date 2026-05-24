import os
from typing import List, Dict, Any
from .bm25 import BM25Okapi
from .dense import DenseRetriever

class HybridRetriever:
    def __init__(self, data_dir: str = "data"):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = BM25Okapi()
        self.data_dir = data_dir
        self.bm25_path = os.path.join(data_dir, "bm25_index.pkl")
        
        # Load existing index if present
        self.sparse_retriever.load(self.bm25_path)

    def index_documents(self, chunks: List[Dict[str, Any]]):
        """
        Indexes chunks into both the dense and sparse retrievers.
        """
        self.dense_retriever.insert_chunks(chunks)
        
        # Merge with existing documents and refit
        existing_chunks = self.sparse_retriever.doc_contents
        all_chunks = existing_chunks + chunks
        self.sparse_retriever.fit(all_chunks)
        
        # Save state to disk
        self.sparse_retriever.save(self.bm25_path)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Executes dual-path retrieval and applies Reciprocal Rank Fusion (RRF).
        """
        # Path A: Dense Retrieval
        dense_results = self.dense_retriever.search(query, top_k=top_k)
        
        # Path B: Sparse Retrieval
        sparse_results = self.sparse_retriever.search(query, top_k=top_k)
        
        # Fusion: RRF
        return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)

    def _reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], top_k: int, k: int = 60) -> List[Dict]:
        """
        Implements Reciprocal Rank Fusion to merge two ranked lists.
        score = 1 / (k + rank)
        """
        rrf_scores = {}
        chunk_map = {}
        
        # Process dense results
        for rank, doc in enumerate(dense_results):
            cid = doc["chunk_id"]
            chunk_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            
        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            cid = doc["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            
        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Return top_k
        fused_results = []
        for cid in sorted_cids[:top_k]:
            res = chunk_map[cid]
            res["rrf_score"] = rrf_scores[cid]
            fused_results.append(res)
            
        return fused_results
