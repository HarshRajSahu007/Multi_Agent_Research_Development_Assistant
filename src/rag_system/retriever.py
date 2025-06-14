from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .vector_store import VectorStoreManager
from .embedder import MultiModalEmbedder
import logging


class CrossModalRetriever:
    """Advanced retrieval system supporting cross-modal queries."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.vector_store = VectorStoreManager(config)
        self.embedder = MultiModalEmbedder(config)
    
    async def retrieve_relevant_content(
        self, 
        query: str, 
        content_types: List[str] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve relevant content across all modalities."""
        
        # Search vector store
        search_results = await self.vector_store.search(
            query=query,
            collection_types=content_types,
            n_results=max_results
        )
        
        # Filter by similarity threshold if distances are available
        filtered_results = {}
        for content_type, results in search_results.items():
            filtered_results[content_type] = [
                result for result in results
                if result.get("distance") is None or result["distance"] < (1 - similarity_threshold)
            ]
        
        return filtered_results
    
    async def retrieve_contextual_content(
        self,
        query: str,
        context_documents: List[str] = None,
        expand_search: bool = True
    ) -> Dict[str, Any]:
        """Retrieve content with contextual understanding."""
        
        # Base retrieval
        base_results = await self.retrieve_relevant_content(query)
        
        # If we have context documents, expand search
        if expand_search and context_documents:
            expanded_results = await self._expand_contextual_search(query, context_documents)
            base_results = self._merge_results(base_results, expanded_results)
        
        # Rank and organize results
        organized_results = self._organize_results(base_results)
        
        return organized_results
    
    async def _expand_contextual_search(
        self,
        query: str,
        context_documents: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Expand search based on context documents."""
        
        expanded_results = {}
        
        # Generate related queries from context
        related_queries = self._generate_related_queries(query, context_documents)
        
        for related_query in related_queries:
            results = await self.vector_store.search(related_query, n_results=5)
            expanded_results = self._merge_results(expanded_results, results)
        
        return expanded_results
    
    def _generate_related_queries(self, query: str, context_documents: List[str]) -> List[str]:
        """Generate related queries based on context."""
        # Simple implementation - could be enhanced with LLM
        related_queries = []
        
        # Extract key terms from context
        key_terms = []
        for doc in context_documents[:2]:  # Limit to first 2 documents
            words = doc.lower().split()
            # Simple keyword extraction
            important_words = [word for word in words if len(word) > 5 and word.isalpha()]
            key_terms.extend(important_words[:5])
        
        # Generate variations of the query with key terms
        for term in key_terms[:3]:
            related_queries.append(f"{query} {term}")
            related_queries.append(f"{term} {query}")
        
        return related_queries
    
    def _merge_results(
        self,
        results1: Dict[str, List[Dict[str, Any]]],
        results2: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Merge two result sets, removing duplicates."""
        merged = results1.copy()
        
        for content_type, items in results2.items():
            if content_type not in merged:
                merged[content_type] = []
            
            # Add items not already present
            existing_ids = {item["id"] for item in merged[content_type]}
            for item in items:
                if item["id"] not in existing_ids:
                    merged[content_type].append(item)
        
        return merged
    
    def _organize_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Organize and rank results by relevance."""
        organized = {
            "summary": {},
            "content": results,
            "recommendations": []
        }
        
        # Generate summary statistics
        total_results = sum(len(items) for items in results.values())
        organized["summary"] = {
            "total_results": total_results,
            "content_types": list(results.keys()),
            "type_counts": {k: len(v) for k, v in results.items()}
        }
        
        # Generate recommendations based on result patterns
        if results.get("figures") and results.get("tables"):
            organized["recommendations"].append("Consider examining both figures and tables for comprehensive data analysis")
        
        if results.get("equations") and results.get("documents"):
            organized["recommendations"].append("Mathematical content found - consider implementation possibilities")
        
        return organized