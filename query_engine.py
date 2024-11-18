import sys
sys.path.append('/Users/krishnaarora/Desktop/content-engine/backend')

from generate_embeddings import EmbeddingGenerator
from vector_store import VectorStore

class QueryEngine:
    def __init__(self):
        print("Initializing QueryEngine...")
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()

    def process_query(self, query):
        """
        Process user queries to find relevant information.
        """
        try:
            print(f"Processing query: {query}")
            
            # Ensure the query is converted to the correct format (1D array)
            query_embedding = self.embedding_generator.model.encode([query]).flatten()
            print(f"Query embedding generated: {query_embedding}")
            
            # Retrieve top K similar results
            results = self.vector_store.retrieve_similar(query_embedding)
            print(f"Results retrieved: {results}")
            
            if not results or len(results) == 0:
                return "No relevant documents found."
            
            return results
        
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

# Example query
if __name__ == "__main__":
    query_engine = QueryEngine()
    query = "What is Chroma?"  # Example query
    print("Query engine initialized. Processing query...")
    results = query_engine.process_query(query)
    print(f"Final results: {results}")
