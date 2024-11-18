import chromadb

class VectorStore:
    def __init__(self, db_path="chromadb_store"):
        # New initialization for the chromadb client
        self.client = chromadb.Client()

        # Create or retrieve the collection
        if "document_embeddings" not in self.client.list_collections():
            self.collection = self.client.create_collection(name="document_embeddings")
            print("Created a new collection: document_embeddings.")
        else:
            self.collection = self.client.get_collection(name="document_embeddings")
            print("Using existing collection: document_embeddings.")

    def store_embeddings(self, chunks, embeddings):
        """
        Store text chunks and their embeddings in the vector store.
        """
        print("Storing embeddings...")
        for idx, (chunk, embed) in enumerate(zip(chunks, embeddings)):
            if not isinstance(embed, list) or not all(isinstance(i, (int, float)) for i in embed):
                raise ValueError("Each embedding should be a list of numbers (int or float).")
            
            # Generate a unique ID for each document (can be the index or any string)
            doc_id = str(idx)  # You can customize this as needed
            
            # Add the document with its ID, metadata, and embedding
            self.collection.add(
                documents=[chunk],
                metadatas={"doc_id": doc_id},
                embeddings=[embed],
                ids=[doc_id]  # Ensure you pass a list of IDs
            )
            print(f"Stored document {doc_id} with embedding.")

    def retrieve_similar(self, query_embedding, top_k=5):
        """
        Retrieve top_k similar documents to the query embedding.
        """
        print(f"Querying for top {top_k} similar documents...")
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        if not results["documents"]:
            print("No similar documents found.")
            return []
        print("Found similar documents:")
        return results["documents"]

# Example usage
if __name__ == "__main__":
    # Create a VectorStore instance
    vector_store = VectorStore()

    # Sample data to store
    sample_chunks = ["This is a sample document.", "Another document with different content."]
    sample_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Example embeddings

    # Store the sample data
    vector_store.store_embeddings(sample_chunks, sample_embeddings)

    # Query with a sample embedding
    query_embedding = [0.2, 0.3, 0.4]
    similar_docs = vector_store.retrieve_similar(query_embedding)
    print(f"Similar documents: {similar_docs}")