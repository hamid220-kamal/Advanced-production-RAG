from transformers import pipeline
import logging

class LocalLLM:
    def __init__(self, model_name="google/flan-t5-small"):
        """
        Initializes an ultra-lightweight T5 model for local text generation
        and Hypothetical Document Embeddings (HyDE).
        """
        logging.info(f"Loading local generator: {model_name}")
        self.generator = pipeline("text2text-generation", model=model_name)

    def generate_hyde(self, query: str) -> str:
        """
        Generates a hypothetical document based on the query to improve dense retrieval matching.
        """
        prompt = f"Please write a short informational paragraph answering the following question: {query}\nAnswer:"
        # T5 is good at sequence to sequence. 
        result = self.generator(prompt, max_length=150, num_return_sequences=1)
        hypothetical_doc = result[0]['generated_text']
        return hypothetical_doc

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates a final answer to the user query based purely on the provided context.
        """
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        result = self.generator(prompt, max_length=200, num_return_sequences=1)
        return result[0]['generated_text']
