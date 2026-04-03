# =============================
# Deep-Dent RAG Gradio App
# =============================

import gradio as gr
import time
import tiktoken

from data_loader import load_documents
from chunking import split_documents
from vector_store import create_vector_store, create_retriever
from generator import create_prompt, create_llm, create_rag_chain
from monitoring import record_request, record_latency, record_token_cost
from guardrails import QueryGuardRails, ResponseGuardRails, RateLimiter
from metafilters import ResponseRanker, ConfidenceCalculator, ContextValidator

rate_limiter = RateLimiter(max_queries_per_minute=30)

# Load pipeline
documents = load_documents("Delivering_better_oral_health.pdf")
chunks = split_documents(documents)
vectorstore = create_vector_store(chunks)
retriever = create_retriever(vectorstore)
prompt = create_prompt()
llm = create_llm()
rag_chain = create_rag_chain(retriever, prompt, llm)

def ask_deep_dent(question):
    """Process dental health question through RAG pipeline with guard rails."""
    try:
        # Rate limiting
        allowed, msg = rate_limiter.is_allowed()
        if not allowed:
            return f"Error: {msg}"
        
        # Input validation
        if not question or not question.strip():
            return "Please provide a valid dental health question."
        
        safe, safety_msg = QueryGuardRails.is_safe_query(question)
        if not safe:
            return f"Query validation failed: {safety_msg}"
        
        is_relevant, _ = QueryGuardRails.is_domain_relevant(question)
        if not is_relevant:
            return "Please ask a question related to dental health."
        
        record_request()
        
        # Get context
        retrieval_results = retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in retrieval_results])
        
        # Validate context
        context_validation = ContextValidator.validate_context(
            [doc.page_content for doc in retrieval_results], question
        )
        
        start_time = time.time()
        response = rag_chain.invoke(question)
        latency = time.time() - start_time
        
        if not response:
            return "Unable to generate response. Please try again."
        
        record_latency(latency)
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = len(enc.encode(response))
        record_token_cost(tokens)
        
        # Add safety disclaimers
        response = ResponseGuardRails.add_medical_disclaimer(response)
        
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

# UI
app = gr.Interface(
    fn=ask_deep_dent,
    inputs=gr.Textbox(lines=3, placeholder="Ask a dental question"),
    outputs=gr.Textbox(lines=10, label="Answer"),
    title="Deep-Dent - Dental Assistant with Guard Rails",
    description="RAG-powered dental health assistant with safety checks"
)

if __name__ == "__main__":
    app.launch(share=True)
