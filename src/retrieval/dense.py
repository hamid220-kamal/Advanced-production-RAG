import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class DenseRetriever:
    def __init__(self, collection_name: str = "rag_collection", embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes ChromaDB for local vector storage.
        Uses a very small 80MB embedding model to save disk space.
        """
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = SentenceTransformer(embedding_model_name)

    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Embeds and inserts chunks into ChromaDB.
        """
        if not chunks:
            return
            
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [{
            "parent_id": chunk["parent_id"],
            "start_index": chunk["start_index"],
            "end_index": chunk["end_index"],
            "source": chunk["source"]
        } for chunk in chunks]
        
        embeddings = self.embedder.encode(documents).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search.
        """
        if self.collection.count() == 0:
            return []
            
        query_embedding = self.embedder.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        parsed_results = []
        if not results["ids"] or not results["ids"][0]:
            return []
            
        for i in range(len(results["ids"][0])):
            parsed_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "parent_id": results["metadatas"][0][i].get("parent_id"),
                "start_index": results["metadatas"][0][i].get("start_index"),
                "end_index": results["metadatas"][0][i].get("end_index"),
                "source": results["metadatas"][0][i].get("source"),
                "dense_distance": results["distances"][0][i]
            })
            
        return parsed_results
