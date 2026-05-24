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

    def rewrite_query(self, current_query: str, chat_history: list) -> str:
        """
        Rewrites the query so it's a standalone search query based on chat history.
        """
        if not chat_history:
            return current_query
            
        history_str = ""
        # Take the last 2 turns
        for turn in chat_history[-2:]:
            history_str += f"{turn['role'].capitalize()}: {turn['content']}\n"
            
        prompt = (
            f"Given the following conversation history and the user's new question, "
            f"rewrite the new question into a standalone query that can be understood without the history.\n\n"
            f"History:\n{history_str}\n"
            f"New question: {current_query}\n"
            f"Standalone Query:"
        )
        result = self.generator(prompt, max_length=100, num_return_sequences=1)
        return result[0]['generated_text']

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

    def stream_answer(self, query: str, context: str):
        """
        Yields the answer token by token for streaming responses.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        streamer = TextIteratorStreamer(self.generator.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(text_inputs=prompt, max_length=200, streamer=streamer)
        thread = Thread(target=self.generator, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
