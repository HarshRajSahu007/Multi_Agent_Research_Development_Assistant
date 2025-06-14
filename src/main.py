import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from document_analysis.document_processor import DocumentProcessor
from rag_system.vector_store import VectorStoreManager
from agent_ecosystem.orchestrator import OrchestratorAgent
from visual_system.visual_parser import VisualParser
from ui.app import ResearchApp
from utils.logger import setup_logger


class MultiAgentResearchSystem:
    """Main system orchestrator for the multi-agent research assistant."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = setup_logger(__name__)
        
        # Initialize core components
        self.document_processor = DocumentProcessor(self.config)
        self.vector_store = VectorStoreManager(self.config)
        self.visual_parser = VisualParser(self.config)
        self.orchestrator = OrchestratorAgent(self.config)
        
        # Initialize data directories
        self._setup_directories()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            "data/embeddings",
            "data/knowledge_graphs", 
            "data/papers",
            "data/experiment_results"
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    async def process_research_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a research query through the multi-agent system."""
        self.logger.info(f"Processing research query: {query}")
        
        # Use orchestrator to coordinate agents
        result = await self.orchestrator.process_query(query, context)
        return result
    
    async def analyze_document(self, document_path: str) -> Dict[str, Any]:
        """Analyze a research document."""
        self.logger.info(f"Analyzing document: {document_path}")
        
        # Process document
        processed_doc = await self.document_processor.process_document(document_path)
        
        # Store in vector database
        await self.vector_store.store_document(processed_doc)
        
        return processed_doc
    
    def run_ui(self):
        """Launch the Streamlit UI."""
        app = ResearchApp(self)
        app.run()


if __name__ == "__main__":
    system = MultiAgentResearchSystem()
    system.run_ui()