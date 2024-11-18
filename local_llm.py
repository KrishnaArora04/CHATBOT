from transformers import pipeline

class LocalLanguageModel:
    def __init__(self, model_name="distilbert-base-uncased"):
        # Initialize the Hugging Face model using the transformers pipeline for text generation
        self.llm = pipeline("text-generation", model=model_name)

    def generate_response(self, query, context):
        """
        Generate a response using the local LLM.
        """
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = self.llm(prompt, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']
