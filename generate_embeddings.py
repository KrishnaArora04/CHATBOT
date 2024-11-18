from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Initializing SentenceTransformer with model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text):
        """
        Generate embeddings for text chunks.
        """
        print("Generating embeddings...")
        chunks = text.split("\n\n")  # Split text into chunks (based on paragraphs or sections)
        embeddings = self.model.encode(chunks)
        print(f"Generated {len(embeddings)} embeddings.")
        return embeddings, chunks

# Test the EmbeddingGenerator
if __name__ == "__main__":
    # Example text to test embedding generation
    test_text = "This is a test document. \n\nThis is another paragraph."
    
    embedding_generator = EmbeddingGenerator()
    embeddings, chunks = embedding_generator.generate_embeddings(test_text)
    
    print("Chunks of text:")
    for chunk in chunks:
        print(chunk)
    
    print("Embedding output:")
    print(embeddings)
