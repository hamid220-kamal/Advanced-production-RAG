import re
import nltk
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import uuid

# Download punkt and punkt_tab for sentence tokenization if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class SemanticChunker:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", percentile_threshold: float = 85.0):
        """
        Initializes the Semantic Chunker.
        Uses a lightweight embedding model (under 100MB) to save local space.
        """
        self.embedder = SentenceTransformer(embedding_model_name)
        self.percentile_threshold = percentile_threshold

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits raw text into individual sentences using NLTK."""
        sentences = nltk.tokenize.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def _combine_sentences(self, sentences: List[str], buffer_size: int = 1) -> List[dict]:
        """Combines sentences into small buffered windows for context-aware embedding."""
        combined_sentences = []
        for i in range(len(sentences)):
            start = max(0, i - buffer_size)
            end = min(len(sentences), i + 1 + buffer_size)
            combined = ' '.join(sentences[start:end])
            combined_sentences.append({
                "sentence": sentences[i],
                "combined_sentence": combined,
                "index": i
            })
        return combined_sentences

    def chunk_document(self, text: str, source: str = "unknown") -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        Splits text into chunks based on semantic boundaries (cosine distance > percentile).
        Returns the chunks and a parent-child mapping.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [], {}
        
        # Buffer sentences for robust embeddings
        buffered_sentences = self._combine_sentences(sentences)
        
        # Get embeddings for the combined sentences
        embeddings = self.embedder.encode([s["combined_sentence"] for s in buffered_sentences])
        
        # Calculate cosine distances between consecutive sentences
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            distance = 1.0 - sim
            distances.append(distance)
            
        # Determine the threshold distance
        if not distances:
            breakpoint_distance = 0.0
        else:
            breakpoint_distance = np.percentile(distances, self.percentile_threshold)
            
        # Identify breakpoint indices
        breakpoints = [i for i, d in enumerate(distances) if d > breakpoint_distance]
        
        # Create chunks
        chunks = []
        parent_id = str(uuid.uuid4())  # Track the whole document as parent
        parent_child_mapping = {parent_id: []}
        
        start_idx = 0
        for bp in breakpoints:
            chunk_sentences = sentences[start_idx:bp+1]
            chunk_text = ' '.join(chunk_sentences)
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "content": chunk_text,
                "start_index": start_idx,
                "end_index": bp,
                "source": source
            })
            parent_child_mapping[parent_id].append(chunk_id)
            start_idx = bp + 1
            
        # Add the final chunk
        if start_idx < len(sentences):
            chunk_sentences = sentences[start_idx:]
            chunk_text = ' '.join(chunk_sentences)
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "content": chunk_text,
                "start_index": start_idx,
                "end_index": len(sentences) - 1,
                "source": source
            })
            parent_child_mapping[parent_id].append(chunk_id)
            
        return chunks, parent_child_mapping
