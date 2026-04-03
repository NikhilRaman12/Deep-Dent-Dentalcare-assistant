"""
Deep-Dent RAG System - Main Pipeline with Guard Rails
Interview-ready, production-style demo with safety checks
"""

import time
import tiktoken
from data_loader import load_documents
from chunking import split_documents
from vector_store import create_vector_store, create_retriever
from generator import create_prompt, create_llm, create_rag_chain
from monitoring import record_request, record_latency, record_token_cost
from guardrails import QueryGuardRails, ResponseGuardRails, RetrievalGuardRails, RateLimiter
from metafilters import ResponseRanker, ContextValidator, ConfidenceCalculator, QueryClassifier

rate_limiter = RateLimiter(max_queries_per_minute=20)

def setup_rag_pipeline():
    """Initialize RAG pipeline."""
    print("Loading documents...")
    documents = load_documents("Delivering_better_oral_health.pdf")
    
    print(f"Loaded {len(documents)} pages")
    print("Chunking documents...")
    chunks = split_documents(documents)
    
    print(f"Created {len(chunks)} chunks")
    print("Creating vector store...")
    vectorstore = create_vector_store(chunks)
    
    retriever = create_retriever(vectorstore)
    prompt = create_prompt()
    llm = create_llm()
    rag_chain = create_rag_chain(retriever, prompt, llm)
    
    return rag_chain

def process_query_with_guardrails(rag_chain, question):
    """Process query through RAG pipeline with guard rails and meta-filters."""
    
    # 1. RATE LIMITING
    allowed, rate_msg = rate_limiter.is_allowed()
    if not allowed:
        return None, {"status": "error", "message": rate_msg}
    
    # 2. QUERY VALIDATION
    safe, safety_msg = QueryGuardRails.is_safe_query(question)
    if not safe:
        return None, {"status": "blocked", "reason": safety_msg}
    
    # 3. DOMAIN RELEVANCE CHECK
    is_relevant, relevance_info = QueryGuardRails.is_domain_relevant(question)
    if not is_relevant:
        return None, {"status": "out_of_domain", "reason": "Query not related to dental health"}
    
    # 4. QUERY CLASSIFICATION
    query_type = QueryClassifier.classify(question)
    
    record_request()
    
    # 5. RETRIEVE CONTEXT
    try:
        # Get raw retrieval results
        retrieval_results = rag_chain.retriever.invoke(question)
        context_text = "\n".join([doc.page_content for doc in retrieval_results])
        
        # Validate retrieval quality
        retrieval_quality = RetrievalGuardRails.validate_retrieval_quality(question, retrieval_results)
        context_validation = ContextValidator.validate_context([doc.page_content for doc in retrieval_results], question)
        
        if not context_validation["valid"]:
            return None, {"status": "insufficient_context", "reason": context_validation["issues"][0]}
    
    except Exception as e:
        return None, {"status": "retrieval_error", "message": str(e)}
    
    # 6. GENERATE RESPONSE
    start_time = time.time()
    try:
        response = rag_chain.invoke(question)
    except Exception as e:
        return None, {"status": "generation_error", "message": str(e)}
    
    latency = time.time() - start_time
    record_latency(latency)
    
    # 7. RESPONSE VALIDATION
    is_valid, validity_msg = ResponseGuardRails.is_valid_response(response)
    if not is_valid:
        return None, {"status": "invalid_response", "reason": validity_msg}
    
    # 8. HALLUCINATION CHECK
    hallucination_risk, hallucination_info = ResponseGuardRails.check_hallucination_risk(response, context_text)
    
    # 9. RESPONSE SCORING & RANKING
    response_scores = ResponseRanker.score_response(question, response, context_text)
    
    # 10. CONFIDENCE CALCULATION
    confidence_data = ConfidenceCalculator.calculate_confidence(
        response_scores,
        context_validation,
        {"quality_ratio": len(retrieval_results) / max(len(retrieval_results), 1)}
    )
    
    # 11. ADD SAFETY DISCLAIMERS
    response_with_disclaimer = ResponseGuardRails.add_confidence_disclaimer(response, confidence_data["overall_confidence"])
    response_with_disclaimer = ResponseGuardRails.add_medical_disclaimer(response_with_disclaimer)
    
    # 12. TOKEN COUNTING
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(response_with_disclaimer))
    record_token_cost(tokens)
    
    # 13. BUILD METADATA
    metadata = {
        "status": "success",
        "latency": latency,
        "tokens": tokens,
        "query_type": query_type["primary_type"],
        "hallucination_risk": hallucination_risk,
        "confidence": confidence_data["overall_confidence"],
        "confidence_level": confidence_data["confidence_level"],
        "context_chunks": len(retrieval_results),
        "response_score": response_scores["overall_score"]
    }
    
    return response_with_disclaimer, metadata

def main():
    """Main demo script."""
    print("=" * 60)
    print("Deep-Dent RAG Assistant - With Guard Rails & Meta-Filters")
    print("=" * 60)
    
    try:
        rag_chain = setup_rag_pipeline()
    except Exception as e:
        print(f"Failed to setup pipeline: {str(e)}")
        return
    
    test_queries = [
        "What are the most effective practices to protect teeth from decay?",
        "How often should I visit the dentist?",
        "What is the importance of fluoride?",
    ]
    
    print("\n" + "=" * 60)
    print("Running Guarded Demo Queries")
    print("=" * 60)
    
    for question in test_queries:
        try:
            print(f"\nQ: {question}")
            response, metadata = process_query_with_guardrails(rag_chain, question)
            
            if metadata["status"] == "success":
                print(f"A: {response[:120]}...")
                print(f"Latency: {metadata['latency']:.2f}s | Tokens: {metadata['tokens']}")
                print(f"Confidence: {metadata['confidence_level']} ({metadata['confidence']:.2f})")
                print(f"Hallucination Risk: {metadata['hallucination_risk']:.2f}")
                print(f"Query Type: {metadata['query_type']}")
            else:
                print(f"Status: {metadata['status']}")
                print(f"Reason: {metadata.get('reason', metadata.get('message', 'Unknown'))}")
        
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()