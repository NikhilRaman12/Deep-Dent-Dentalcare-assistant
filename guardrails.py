"""
Guard Rails for RAG System
Input validation, confidence checks, safety filters
"""

import re
from typing import Tuple, Dict

class QueryGuardRails:
    """Validate and sanitize user queries."""
    
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 500
    BLOCKED_KEYWORDS = ['sql injection',  'ignore', 'previous' , 'instructions', drop table', '<script>', '--', '/*']
    
    @staticmethod
    def is_safe_query(query: str) -> Tuple[bool, str]:
        """Check if query is safe to process."""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) < QueryGuardRails.MIN_QUERY_LENGTH:
            return False, "Query too short (min 3 characters)"
        
        if len(query) > QueryGuardRails.MAX_QUERY_LENGTH:
            return False, "Query too long (max 500 characters)"
        
        query_lower = query.lower()
        for keyword in QueryGuardRails.BLOCKED_KEYWORDS:
            if keyword in query_lower:
                return False, f"Query contains blocked content: {keyword}"
        
        return True, "Safe"
    
    @staticmethod
    def is_domain_relevant(query: str) -> Tuple[bool, Dict]:
        """Check if query is relevant to dental domain."""
        dental_keywords = ['tooth', 'teeth', 'dental', 'dentist', 'cavity', 'decay', 
                          'brush', 'floss', 'gum', 'enamel', 'crown', 'filling', 
                          'implant', 'orthodont', 'root canal', 'plaque', 'tartar',
                          'whitening', 'cleaning', 'checkup', 'fluoride']
        
        query_lower = query.lower()
        found_keywords = [kw for kw in dental_keywords if kw in query_lower]
        
        relevance_score = len(found_keywords) / len(dental_keywords)
        is_relevant = relevance_score >= 0.1  # At least 10% match
        
        return is_relevant, {
            "score": relevance_score,
            "keywords_found": found_keywords,
            "is_dental_query": is_relevant
        }

class ResponseGuardRails:
    """Validate and filter generated responses."""
    
    MIN_RESPONSE_LENGTH = 10
    MAX_RESPONSE_LENGTH = 2000
    CONFIDENCE_THRESHOLD = 0.3
    
    @staticmethod
    def is_valid_response(response: str) -> Tuple[bool, str]:
        """Check if response is valid."""
        if not response or not response.strip():
            return False, "Response is empty"
        
        if len(response) < ResponseGuardRails.MIN_RESPONSE_LENGTH:
            return False, "Response too short"
        
        if len(response) > ResponseGuardRails.MAX_RESPONSE_LENGTH:
            return False, "Response too long"
        
        return True, "Valid"
    
    @staticmethod
    def check_hallucination_risk(response: str, context: str) -> Tuple[float, Dict]:
        """Estimate hallucination risk based on response-context overlap."""
        # Simple heuristic: measure word overlap
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        # Filter out stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'is', 'are', 'be'}
        response_words -= stop_words
        context_words -= stop_words
        
        if len(response_words) == 0:
            return 0.5, {"warning": "Empty response after filtering"}
        
        overlap = len(response_words & context_words) / len(response_words)
        hallucination_risk = 1.0 - overlap  # High overlap = low risk
        
        return hallucination_risk, {
            "overlap_ratio": overlap,
            "hallucination_risk": hallucination_risk,
            "is_grounded": hallucination_risk < 0.5
        }
    
    @staticmethod
    def add_confidence_disclaimer(response: str, confidence: float) -> str:
        """Add confidence disclaimer to response."""
        if confidence < 0.5:
            return response + "\n\n[DISCLAIMER: Low confidence in this response. Consult a dental professional.]"
        elif confidence < 0.7:
            return response + "\n\n[NOTE: This information should be verified by a professional.]"
        return response

class RetrievalGuardRails:
    """Validate retrieved context quality."""
    
    RELEVANCE_THRESHOLD = 0.4
    MIN_RETRIEVAL_SCORE = 0.3
    
    @staticmethod
    def filter_low_quality_results(results: list, threshold: float = 0.4) -> Tuple[list, Dict]:
        """Filter out low-quality retrieval results."""
        if not results:
            return [], {"filtered": 0, "total": 0, "kept": 0}
        
        filtered = [r for r in results if hasattr(r, 'metadata') and 
                   r.metadata.get('score', 1.0) >= threshold]
        
        return filtered, {
            "total_retrieved": len(results),
            "kept": len(filtered),
            "filtered": len(results) - len(filtered),
            "quality_ratio": len(filtered) / len(results) if results else 0
        }
    
    @staticmethod
    def validate_retrieval_quality(query: str, context_chunks: list) -> Dict:
        """Validate overall retrieval quality."""
        if not context_chunks:
            return {"is_valid": False, "reason": "No relevant context retrieved"}
        
        total_context_length = sum(len(str(chunk)) for chunk in context_chunks)
        
        if total_context_length < 100:
            return {"is_valid": False, "reason": "Context too sparse"}
        
        return {
            "is_valid": True,
            "chunks_retrieved": len(context_chunks),
            "total_context_length": total_context_length,
            "avg_chunk_size": total_context_length / len(context_chunks)
        }

class RateLimiter:
    """Simple rate limiting to prevent abuse."""
    
    def __init__(self, max_queries_per_minute: int = 20):
        self.max_queries = max_queries_per_minute
        self.query_timestamps = []
    
    def is_allowed(self) -> Tuple[bool, str]:
        """Check if query is allowed under rate limit."""
        import time
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.query_timestamps = [ts for ts in self.query_timestamps 
                                if current_time - ts < 60]
        
        if len(self.query_timestamps) >= self.max_queries:
            return False, f"Rate limit exceeded: {self.max_queries} queries/minute"
        
        self.query_timestamps.append(current_time)
        return True, "Allowed"

class ResponseSafetyFilter:
    """Filter responses for medical safety."""
    
    EMERGENCY_KEYWORDS = ['emergency', 'hospital', 'urgent', 'severe', 'danger']
    
    @staticmethod
    def contains_emergency_indicator(response: str) -> bool:
        """Check if response warrants emergency referral."""
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in ResponseSafetyFilter.EMERGENCY_KEYWORDS)
    
    @staticmethod
    def add_medical_disclaimer(response: str) -> str:
        """Add standard medical disclaimer."""
        disclaimer = "\n\n[IMPORTANT: This information is for educational purposes only. " \
                    "Always consult a licensed dentist for professional medical advice.]"
        if disclaimer not in response:
            return response + disclaimer
        return response
