"""
Simple RAG Evaluation Metrics
- Precision@K: % of retrieved docs that are relevant
- Recall: % of relevant docs that were retrieved
- Latency: Response time in seconds
- Token cost: Total tokens generated
"""

import time
import tiktoken
from typing import List, Tuple

def calculate_precision_k(retrieved_docs, relevant_indices, k=4):
    """Calculate precision@k."""
    relevant_count = sum(1 for i in range(min(k, len(retrieved_docs))) 
                        if i in relevant_indices)
    return relevant_count / k if k > 0 else 0

def calculate_recall(retrieved_indices, relevant_indices):
    """Calculate recall."""
    if len(relevant_indices) == 0:
        return 1.0
    retrieved_set = set(retrieved_indices[:4])
    relevant_set = set(relevant_indices)
    return len(retrieved_set & relevant_set) / len(relevant_set)

def evaluate_response(response_text):
    """Calculate metrics for a response."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(response_text))
    return tokens

def benchmark_rag(rag_chain, queries: List[str], latency_threshold=2.0):
    """Benchmark RAG performance."""
    metrics = {
        "total_queries": len(queries),
        "avg_latency": 0,
        "total_tokens": 0,
        "within_threshold": 0
    }
    
    latencies = []
    
    for query in queries:
        start = time.time()
        response = rag_chain.invoke(query)
        latency = time.time() - start
        
        latencies.append(latency)
        tokens = evaluate_response(response)
        
        metrics["total_tokens"] += tokens
        if latency <= latency_threshold:
            metrics["within_threshold"] += 1
    
    metrics["avg_latency"] = sum(latencies) / len(latencies) if latencies else 0
    metrics["within_threshold_pct"] = (metrics["within_threshold"] / len(queries)) * 100
    
    return metrics