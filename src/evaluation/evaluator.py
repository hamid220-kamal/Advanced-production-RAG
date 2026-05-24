from sentence_transformers import CrossEncoder

class LocalEvaluator:
    def __init__(self, cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Uses the existing CrossEncoder to calculate Answer Relevancy and Context Precision.
        """
        self.ce_model = CrossEncoder(cross_encoder_model, max_length=512)

    def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """
        Evaluates how well the generated answer addresses the query.
        """
        score = self.ce_model.predict([query, answer])
        return float(score)

    def evaluate_context_precision(self, query: str, retrieved_chunks: list) -> float:
        """
        Evaluates the relevance of the retrieved chunks to the query.
        Returns the average cross-encoder score for the top k chunks.
        """
        if not retrieved_chunks:
            return 0.0
            
        pairs = [[query, chunk["content"]] for chunk in retrieved_chunks]
        scores = self.ce_model.predict(pairs)
        
        avg_score = sum(scores) / len(scores)
        return float(avg_score)
