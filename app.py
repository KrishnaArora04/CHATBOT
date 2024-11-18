import sys
import streamlit as st

# Add backend folder to sys.path
sys.path.append('/Users/krishnaarora/Desktop/content-engine/backend')

# Import modules
from parse_documents import extract_text_from_pdf
from query_engine import QueryEngine
from local_llm import LocalLanguageModel

# Paths to the PDF files
pdf_paths = [
    "/Users/krishnaarora/Desktop/content-engine/data/goog-10-k-2023 (1).pdf",
    "/Users/krishnaarora/Desktop/content-engine/data/tsla-20231231-gen.pdf",
    "/Users/krishnaarora/Desktop/content-engine/data/uber-10-k-2023.pdf"
]

# Extract text from PDFs
st.title("Content Engine Chatbot")
st.write("Analyzing and comparing Form 10-K filings from Alphabet, Tesla, and Uber.")

# Initialize components
query_engine = QueryEngine()
llm = LocalLanguageModel()

# Process PDFs
document_texts = []
for path in pdf_paths:
    try:
        text = extract_text_from_pdf(path)
        document_texts.append(text)
        st.write(f"Extracted text from {path[:50]}...: {text[:200]}")  # Debugging
    except Exception as e:
        st.error(f"Error extracting text from {path}: {e}")

if any(isinstance(text, str) and text.startswith("Error") for text in document_texts):
    st.error("There was an issue extracting text from one of the PDFs. Please check the files.")
else:
    # Button for indexing documents
    if st.button("Index Documents"):
        try:
            for text in document_texts:
                embeddings, chunks = query_engine.embedding_generator.generate_embeddings(text)
                query_engine.vector_store.store_embeddings(chunks, embeddings)
                st.write(f"Document indexed: {text[:200]}")  # Debugging
            st.success("Documents indexed successfully!")
        except Exception as e:
            st.error(f"Error during indexing: {e}")

# User Query
query = st.text_input("Ask a question about the documents:")
if query:
    try:
        results = query_engine.process_query(query)
        st.write("Query Results (raw):", results)  # Debugging

        if isinstance(results, str) and "Error" in results:
            st.error(results)
        elif results:
            context = " ".join([res["document"] for res in results])
            st.write("Context used for response:", context[:500])  # Debugging
            response = llm.generate_response(query, context)
            st.subheader("Query Results")
            st.write(context)
            st.subheader("Response")
            st.write(response)
    except Exception as e:
        st.error(f"Error processing query: {e}")
