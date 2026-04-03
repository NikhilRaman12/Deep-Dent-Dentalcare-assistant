from langchain_community.document_loaders import PyPDFLoader

def load_documents(pdf_path: str):
    """Load PDF documents."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError(f"No content found in {pdf_path}")
        return docs
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    except Exception as e:
        raise Exception(f"Error loading PDF: {str(e)}")