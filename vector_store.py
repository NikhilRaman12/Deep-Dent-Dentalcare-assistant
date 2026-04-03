from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    """Create FAISS vector store from document chunks."""
    if not chunks:
        raise ValueError("No chunks provided to create vector store")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        raise Exception(f"Error creating vector store: {str(e)}")

def create_retriever(vectorstore, k=4):
    """Create retriever to fetch top-k relevant documents."""
    if not vectorstore:
        raise ValueError("Vector store is required")
    
    try:
        return vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        raise Exception(f"Error creating retriever: {str(e)}")