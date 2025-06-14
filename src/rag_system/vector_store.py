import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import json
from sentence_transformers import SentenceTransformer


class VectorStoreManager:
    """Manages vector storage and retrieval for multimodal content."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config["databases"]["vector_store"]["persist_directory"]
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config["models"]["embedding_model"])
        
        # Create collections for different content types
        self.collections = {
            "documents": self._get_or_create_collection("documents"),
            "figures": self._get_or_create_collection("figures"),
            "tables": self._get_or_create_collection("tables"), 
            "equations": self._get_or_create_collection("equations"),
            "code": self._get_or_create_collection("code")
        }
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(name)
    
    async def store_document(self, processed_doc: Dict[str, Any]) -> bool:
        """Store processed document in vector database."""
        try:
            doc_id = self._generate_doc_id(processed_doc["path"])
            
            # Store main document content
            doc_text = processed_doc.get("title", "") + " " + processed_doc.get("abstract", "")
            if doc_text.strip():
                doc_embedding = self.embedding_model.encode([doc_text])[0]
                
                self.collections["documents"].add(
                    embeddings=[doc_embedding.tolist()],
                    documents=[doc_text],
                    metadatas=[{
                        "path": processed_doc["path"],
                        "title": processed_doc.get("title", ""),
                        "type": "document"
                    }],
                    ids=[doc_id]
                )
            
            # Store figures
            for i, figure in enumerate(processed_doc.get("figures", [])):
                await self._store_figure(figure, doc_id, i)
            
            # Store tables
            for i, table in enumerate(processed_doc.get("tables", [])):
                await self._store_table(table, doc_id, i)
            
            # Store equations
            for i, equation in enumerate(processed_doc.get("equations", [])):
                await self._store_equation(equation, doc_id, i)
            
            self.logger.info(f"Successfully stored document: {processed_doc['path']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store document: {e}")
            return False
    
    async def _store_figure(self, figure: Dict[str, Any], doc_id: str, figure_index: int):
        """Store figure information in vector database."""
        # Create text description of figure for embedding
        analysis = figure.get("analysis", {})
        description = f"Figure {figure_index}: "
        
        if analysis.get("is_chart"):
            description += "Chart/Graph "
        if analysis.get("is_diagram"):
            description += "Diagram "
        
        description += f"Dimensions: {analysis.get('dimensions', {}).get('width', 0)}x{analysis.get('dimensions', {}).get('height', 0)} "
        description += f"Complexity: {analysis.get('complexity', 'unknown')}"
        
        embedding = self.embedding_model.encode([description])[0]
        
        self.collections["figures"].add(
            embeddings=[embedding.tolist()],
            documents=[description],
            metadatas=[{
                "doc_id": doc_id,
                "page": figure.get("page", 0),
                "index": figure_index,
                "type": "figure",
                "analysis": json.dumps(analysis)
            }],
            ids=[f"{doc_id}_figure_{figure_index}"]
        )
    
    async def _store_table(self, table: Dict[str, Any], doc_id: str, table_index: int):
        """Store table information in vector database."""
        # Create text representation of table
        headers = table.get("headers", [])
        shape = table.get("shape", [0, 0])
        
        description = f"Table {table_index}: {len(headers)} columns ({', '.join(headers[:3])}) "
        description += f"Shape: {shape[0]} rows x {shape[1]} columns"
        
        # Add some data samples
        data = table.get("data", [])
        if data:
            first_row = data[0]
            sample_data = " ".join([f"{k}: {v}" for k, v in list(first_row.items())[:3]])
            description += f" Sample: {sample_data}"
        
        embedding = self.embedding_model.encode([description])[0]
        
        self.collections["tables"].add(
            embeddings=[embedding.tolist()],
            documents=[description],
            metadatas=[{
                "doc_id": doc_id,
                "page": table.get("page", 0),
                "index": table_index,
                "type": "table",
                "headers": json.dumps(headers),
                "shape": json.dumps(shape)
            }],
            ids=[f"{doc_id}_table_{table_index}"]
        )
    
    async def _store_equation(self, equation: Dict[str, Any], doc_id: str, eq_index: int):
        """Store equation information in vector database."""
        content = equation.get("content", "")
        eq_type = equation.get("type", "unknown")
        analysis = equation.get("analysis", {})
        
        description = f"Equation {eq_index} ({eq_type}): {content} "
        description += f"Complexity: {analysis.get('complexity', 'unknown')}"
        
        embedding = self.embedding_model.encode([description])[0]
        
        self.collections["equations"].add(
            embeddings=[embedding.tolist()],
            documents=[description],
            metadatas=[{
                "doc_id": doc_id,
                "page": equation.get("page", 0),
                "index": eq_index,
                "type": "equation",
                "equation_type": eq_type,
                "content": content
            }],
            ids=[f"{doc_id}_equation_{eq_index}"]
        )
    
    def _generate_doc_id(self, doc_path: str) -> str:
        """Generate unique document ID from path."""
        import hashlib
        return hashlib.md5(doc_path.encode()).hexdigest()
    
    async def search(self, query: str, collection_types: List[str] = None, n_results: int = 10) -> Dict[str, List[Dict]]:
        """Search across collections for relevant content."""
        if collection_types is None:
            collection_types = list(self.collections.keys())
        
        query_embedding = self.embedding_model.encode([query])[0]
        results = {}
        
        for collection_type in collection_types:
            if collection_type in self.collections:
                try:
                    collection_results = self.collections[collection_type].query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=n_results
                    )
                    
                    # Format results
                    formatted_results = []
                    for i in range(len(collection_results["ids"][0])):
                        formatted_results.append({
                            "id": collection_results["ids"][0][i],
                            "document": collection_results["documents"][0][i],
                            "metadata": collection_results["metadatas"][0][i],
                            "distance": collection_results["distances"][0][i] if "distances" in collection_results else None
                        })
                    
                    results[collection_type] = formatted_results
                    
                except Exception as e:
                    self.logger.error(f"Search failed for collection {collection_type}: {e}")
                    results[collection_type] = []
        
        return results