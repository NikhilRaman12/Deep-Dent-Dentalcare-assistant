"""
Meta Filters for RAG System
Response ranking, quality scoring, context validation
"""

from typing import Dict, List, Tuple
import re

class ResponseRanker:
    """Rank and score responses by quality."""
    
    @staticmethod
    def score_response(query: str, response: str, context: str) -> Dict:
        """Calculate response quality score (0-1)."""
        scores = {}
        
        # Length score: prefer moderate length responses
        response_length = len(response.split())
        length_score = min(response_length / 150, 1.0)  # Optimal ~150 words
        scores["length_score"] = length_score
        
        # Relevance score: query terms in response
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        relevance = len(query_terms & response_terms) / len(query_terms) if query_terms else 0
        scores["relevance_score"] = min(relevance, 1.0)
        
        # Context grounding: response overlap with context
        context_terms = set(context.lower().split())
        grounding = len(response_terms & context_terms) / len(response_terms) if response_terms else 0
        scores["grounding_score"] = grounding
        
        # Specificity score: avoid vague responses
        vague_words = ['maybe', 'perhaps', 'could', 'might', 'possibly', 'generally']
        vague_count = sum(1 for word in vague_words if word in response.lower())
        specificity_score = 1.0 - (vague_count * 0.1)
        scores["specificity_score"] = max(specificity_score, 0.0)
        
        # Sentence structure: proper punctuation
        sentences = response.split('.')
        proper_sentences = sum(1 for s in sentences if len(s.split()) > 3)
        structure_score = proper_sentences / len(sentences) if sentences else 0
        scores["structure_score"] = structure_score
        
        # Overall composite score
        weights = {
            "length_score": 0.15,
            "relevance_score": 0.35,
            "grounding_score": 0.25,
            "specificity_score": 0.15,
            "structure_score": 0.10
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        scores["overall_score"] = overall_score
        
        return scores

class ContextValidator:
    """Validate context quality and relevance."""
    
    @staticmethod
    def validate_context(context_chunks: List[str], query: str) -> Dict:
        """Validate retrieved context."""
        if not context_chunks:
            return {"valid": False, "issue": "No context retrieved"}
        
        issues = []
        
        # Check minimum context size
        total_content = "\n".join(context_chunks)
        if len(total_content) < 50:
            issues.append("Context too sparse")
        
        # Check for duplicate contexts
        unique_chunks = len(set(context_chunks))
        if unique_chunks < len(context_chunks) * 0.8:
            issues.append("High duplication in context")
        
        # Check relevance of context to query
        query_terms = set(query.lower().split())
        context_lower = total_content.lower()
        query_coverage = sum(1 for term in query_terms if term in context_lower) / len(query_terms) if query_terms else 0
        
        if query_coverage < 0.2:
            issues.append("Low query coverage in context")
        
        return {
            "valid": len(issues) == 0,
            "chunks_count": len(context_chunks),
            "unique_chunks": unique_chunks,
            "query_coverage": query_coverage,
            "total_context_size": len(total_content),
            "issues": issues
        }
    
    @staticmethod
    def calculate_context_diversity(context_chunks: List[str]) -> float:
        """Measure diversity of context (0-1)."""
        if len(context_chunks) <= 1:
            return 0.0
        
        # Simple diversity: measure unique terms across chunks
        all_terms = set()
        for chunk in context_chunks:
            all_terms.update(chunk.lower().split())
        
        # Diversity score based on unique terms per chunk
        avg_unique_terms = len(all_terms) / len(context_chunks) if context_chunks else 0
        diversity = min(avg_unique_terms / 20, 1.0)  # Normalize
        
        return diversity

class ConfidenceCalculator:
    """Calculate confidence scores for responses."""
    
    @staticmethod
    def calculate_confidence(response_scores: Dict, context_quality: Dict, 
                           retrieval_quality: Dict) -> Dict:
        """Calculate overall confidence score."""
        
        # Component scores with weights
        response_score = response_scores.get("overall_score", 0.5)
        context_score = 1.0 if not context_quality.get("issues") else 0.6
        retrieval_score = retrieval_quality.get("quality_ratio", 0.5)
        
        weights = {
            "response": 0.4,
            "context": 0.3,
            "retrieval": 0.3
        }
        
        overall_confidence = (
            response_score * weights["response"] +
            context_score * weights["context"] +
            retrieval_score * weights["retrieval"]
        )
        
        return {
            "overall_confidence": overall_confidence,
            "response_quality": response_score,
            "context_quality": context_score,
            "retrieval_quality": retrieval_score,
            "confidence_level": ConfidenceCalculator._get_confidence_label(overall_confidence)
        }
    
    @staticmethod
    def _get_confidence_label(score: float) -> str:
        """Convert confidence score to label."""
        if score >= 0.8:
            return "HIGH"
        elif score >= 0.6:
            return "MEDIUM"
        elif score >= 0.4:
            return "LOW"
        else:
            return "VERY LOW"

class QueryClassifier:
    """Classify query type for better routing."""
    
    QUERY_TYPES = {
        "factual": ["what", "when", "where", "how many", "which"],
        "procedural": ["how", "steps", "process", "way to"],
        "causal": ["why", "because", "cause"],
        "comparison": ["difference", "versus", "vs", "better", "worse"],
        "opinion": ["think", "believe", "should", "best", "worst"]
    }
    
    @staticmethod
    def classify(query: str) -> Dict:
        """Classify query type."""
        query_lower = query.lower()
        
        classifications = {}
        for qtype, keywords in QueryClassifier.QUERY_TYPES.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            classifications[qtype] = score
        
        if not classifications or max(classifications.values()) == 0:
            primary_type = "general"
            confidence = 0.0
        else:
            primary_type = max(classifications, key=classifications.get)
            confidence = min(classifications[primary_type] * 0.3, 1.0)
        
        return {
            "primary_type": primary_type,
            "confidence": confidence,
            "all_types": classifications
        }

class ResponseDeduplicator:
    """Remove duplicate or near-duplicate sentences."""
    
    @staticmethod
    def deduplicate(response: str) -> str:
        """Remove duplicate sentences."""
        sentences = response.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences) + '.' if unique_sentences else response
    
    @staticmethod
    def similarity_score(s1: str, s2: str) -> float:
        """Calculate similarity between two sentences (0-1)."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0