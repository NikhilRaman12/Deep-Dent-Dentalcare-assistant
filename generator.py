from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_prompt():
    """Create prompt template for RAG."""
    return PromptTemplate.from_template(
        """You are Deep-Dent, a dental care assistant.
Answer using ONLY the context provided.

Context: {context}
Question: {question}
Answer:"""
    )

def create_llm():
    """Create local Ollama LLM."""
    try:
        return ChatOllama(model="tinyllama:latest", temperature=0)
    except Exception as e:
        raise Exception(f"Error creating LLM: {str(e)}")

def create_rag_chain(retriever, prompt, llm):
    """Create RAG chain: retriever -> prompt -> LLM -> parser."""
    if not retriever or not prompt or not llm:
        raise ValueError("Retriever, prompt, and LLM are required")
    
    try:
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    except Exception as e:
        raise Exception(f"Error creating RAG chain: {str(e)}")