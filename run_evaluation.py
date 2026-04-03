"""
Run RAG System Evaluation
"""

from data_loader import load_documents
from chunking import split_documents
from vector_store import create_vector_store, create_retriever
from generator import create_prompt, create_llm, create_rag_chain
from evaluation import benchmark_rag

def main():
    try:
        # Setup
        print("Initializing RAG system...")
        documents = load_documents("Delivering_better_oral_health.pdf")
        chunks = split_documents(documents)
        vectorstore = create_vector_store(chunks)
        retriever = create_retriever(vectorstore)
        prompt = create_prompt()
        llm = create_llm()
        rag_chain = create_rag_chain(retriever, prompt, llm)
        print("System initialized successfully.")
        
        # Test queries
        test_queries = [
            "What are the most effective practices to protect teeth from decay?",
            "How often should I visit the dentist?",
            "What is the importance of fluoride in dental health?",
        ]
        
        print("=" * 60)
        print("Deep-Dent RAG System - Evaluation Report")
        print("=" * 60)
        
        # Benchmark
        metrics = benchmark_rag(rag_chain, test_queries, latency_threshold=2.0)
        
        print(f"\nTotal Queries: {metrics['total_queries']}")
        print(f"Avg Latency: {metrics['avg_latency']:.2f}s")
        print(f"Total Tokens: {metrics['total_tokens']}")
        print(f"Within 2s Threshold: {metrics['within_threshold_pct']:.1f}%")
        
        print("\n" + "=" * 60)
    
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    main()