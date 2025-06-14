import networkx as nx
import json
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path


class KnowledgeGraphStore:
    """Manages knowledge graphs for research domain relationships."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.persist_dir = Path(config["databases"]["knowledge_graph"]["persist_directory"])
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize knowledge graphs for different domains
        self.graphs = {
            "concepts": nx.DiGraph(),  # Concept relationships
            "papers": nx.DiGraph(),    # Paper citations and connections
            "authors": nx.DiGraph(),   # Author collaborations
            "methods": nx.DiGraph()    # Methodological relationships
        }
        
        # Load existing graphs
        self._load_graphs()
    
    def add_document_knowledge(self, processed_doc: Dict[str, Any]):
        """Add knowledge from a processed document to the graphs."""
        doc_id = self._generate_doc_id(processed_doc["path"])
        title = processed_doc.get("title", "Untitled")
        
        # Add to papers graph
        self.graphs["papers"].add_node(doc_id, title=title, type="paper")
        
        # Extract and add concepts
        concepts = self._extract_concepts(processed_doc)
        for concept in concepts:
            self._add_concept(concept, doc_id)
        
        # Extract and add methods
        methods = self._extract_methods(processed_doc)
        for method in methods:
            self._add_method(method, doc_id)
    
    def _extract_concepts(self, doc: Dict[str, Any]) -> List[str]:
        """Extract key concepts from document."""
        concepts = []
        
        # Simple keyword extraction from title and abstract
        text = (doc.get("title", "") + " " + doc.get("abstract", "")).lower()
        
        # Common research concepts/keywords
        concept_keywords = [
            "machine learning", "deep learning", "neural network", "classification",
            "regression", "clustering", "optimization", "algorithm", "model",
            "dataset", "training", "validation", "accuracy", "performance",
            "feature", "embedding", "attention", "transformer", "cnn", "rnn",
            "supervised", "unsupervised", "reinforcement", "semi-supervised"
        ]
        
        for keyword in concept_keywords:
            if keyword in text:
                concepts.append(keyword)
        
        return concepts
    
    def _extract_methods(self, doc: Dict[str, Any]) -> List[str]:
        """Extract methodological approaches from document."""
        methods = []
        
        text = (doc.get("title", "") + " " + doc.get("abstract", "")).lower()
        
        # Common methods/approaches
        method_keywords = [
            "gradient descent", "backpropagation", "cross-validation",
            "ensemble", "bagging", "boosting", "svm", "random forest",
            "linear regression", "logistic regression", "k-means",
            "pca", "t-sne", "lstm", "gru", "bert", "gpt"
        ]
        
        for keyword in method_keywords:
            if keyword in text:
                methods.append(keyword)
        
        return methods
    
    def _add_concept(self, concept: str, doc_id: str):
        """Add concept and its relationship to document."""
        if not self.graphs["concepts"].has_node(concept):
            self.graphs["concepts"].add_node(concept, type="concept")
        
        # Add edge from concept to paper
        self.graphs["concepts"].add_edge(concept, doc_id, relationship="mentioned_in")
    
    def _add_method(self, method: str, doc_id: str):
        """Add method and its relationship to document."""
        if not self.graphs["methods"].has_node(method):
            self.graphs["methods"].add_node(method, type="method")
        
        # Add edge from method to paper
        self.graphs["methods"].add_edge(method, doc_id, relationship="used_in")
    
    def find_related_documents(self, doc_id: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        """Find documents related to the given document."""
        related_docs = []
        
        # Find concepts mentioned in this document
        doc_concepts = []
        for graph_name, graph in self.graphs.items():
            if graph.has_node(doc_id):
                # Get neighboring concepts/methods
                neighbors = list(graph.neighbors(doc_id))
                doc_concepts.extend(neighbors)
        
        # Find other documents that share these concepts
        doc_scores = {}
        for concept in doc_concepts:
            for graph_name, graph in self.graphs.items():
                if graph.has_node(concept):
                    concept_neighbors = list(graph.neighbors(concept))
                    for neighbor in concept_neighbors:
                        if neighbor != doc_id and neighbor.startswith('doc_'):
                            doc_scores[neighbor] = doc_scores.get(neighbor, 0) + 1
        
        # Sort by shared concept count
        related_docs = [(doc_id, score) for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]
        
        return related_docs[:10]  # Return top 10
    
    def get_concept_network(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """Get network of related concepts."""
        if concept not in self.graphs["concepts"]:
            return {"nodes": [], "edges": []}
        
        # Use BFS to find related concepts within depth
        subgraph_nodes = set([concept])
        current_level = {concept}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                if node in self.graphs["concepts"]:
                    neighbors = list(self.graphs["concepts"].neighbors(node))
                    next_level.update(neighbors)
            
            subgraph_nodes.update(next_level)
            current_level = next_level
        
        # Create subgraph
        subgraph = self.graphs["concepts"].subgraph(subgraph_nodes)
        
        # Format for visualization
        nodes = [{"id": node, "type": subgraph.nodes[node].get("type", "unknown")} 
                for node in subgraph.nodes()]
        edges = [{"source": edge[0], "target": edge[1], 
                 "relationship": subgraph.edges[edge].get("relationship", "related")} 
                for edge in subgraph.edges()]
        
        return {"nodes": nodes, "edges": edges}
    
    def _generate_doc_id(self, doc_path: str) -> str:
        """Generate unique document ID."""
        import hashlib
        return "doc_" + hashlib.md5(doc_path.encode()).hexdigest()[:10]
    
    def _load_graphs(self):
        """Load saved graphs from disk."""
        for graph_name in self.graphs.keys():
            graph_path = self.persist_dir / f"{graph_name}.json"
            if graph_path.exists():
                try:
                    with open(graph_path, 'r') as f:
                        data = json.load(f)
                    self.graphs[graph_name] = nx.node_link_graph(data)
                    self.logger.info(f"Loaded {graph_name} graph with {len(self.graphs[graph_name].nodes)} nodes")
                except Exception as e:
                    self.logger.error(f"Failed to load {graph_name} graph: {e}")
    
    def save_graphs(self):
        """Save graphs to disk."""
        for graph_name, graph in self.graphs.items():
            graph_path = self.persist_dir / f"{graph_name}.json"
            try:
                data = nx.node_link_data(graph)
                with open(graph_path, 'w') as f:
                    json.dump(data, f, indent=2)
                self.logger.info(f"Saved {graph_name} graph with {len(graph.nodes)} nodes")
            except Exception as e:
                self.logger.error(f"Failed to save {graph_name} graph: {e}")
