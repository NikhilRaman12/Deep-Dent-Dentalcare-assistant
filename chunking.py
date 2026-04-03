from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into manageable chunks."""
    if not documents:
        raise ValueError("No documents provided to chunk")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    
    if not chunks:
        raise ValueError("Failed to create chunks from documents")
    
    return chunks