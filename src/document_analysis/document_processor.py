"""Document analysis components for research papers."""

# src/document_analysis/document_processor.py
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

from .ocr_engine import OCREngine
from .figure_extractor import FigureExtractor
from .table_extractor import TableExtractor
from .equation_parser import EquationParser


class DocumentProcessor:
    """Core document processing engine for research papers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized processors
        self.ocr_engine = OCREngine(config)
        self.figure_extractor = FigureExtractor(config)
        self.table_extractor = TableExtractor(config)
        self.equation_parser = EquationParser(config)
    
    async def process_document(self, document_path: str) -> Dict[str, Any]:
        """Process a research document and extract all components."""
        self.logger.info(f"Processing document: {document_path}")
        
        # Open PDF document
        doc = fitz.open(document_path)
        
        processed_doc = {
            "path": document_path,
            "title": "",
            "abstract": "",
            "text_content": [],
            "figures": [],
            "tables": [],
            "equations": [],
            "metadata": {}
        }
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_data = await self._process_page(page, page_num)
            
            # Aggregate page data
            processed_doc["text_content"].extend(page_data["text"])
            processed_doc["figures"].extend(page_data["figures"])
            processed_doc["tables"].extend(page_data["tables"])
            processed_doc["equations"].extend(page_data["equations"])
        
        # Extract title and abstract
        processed_doc["title"] = self._extract_title(processed_doc["text_content"])
        processed_doc["abstract"] = self._extract_abstract(processed_doc["text_content"])
        
        doc.close()
        return processed_doc
    
    async def _process_page(self, page, page_num: int) -> Dict[str, Any]:
        """Process a single page of the document."""
        page_data = {
            "text": [],
            "figures": [],
            "tables": [],
            "equations": []
        }
        
        # Extract text
        text_content = page.get_text()
        page_data["text"].append({
            "page": page_num,
            "content": text_content,
            "type": "text"
        })
        
        # Extract images/figures
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            figure_data = await self.figure_extractor.extract_figure(page, img, page_num, img_index)
            if figure_data:
                page_data["figures"].append(figure_data)
        
        # Extract tables
        tables = await self.table_extractor.extract_tables(page, page_num)
        page_data["tables"].extend(tables)
        
        # Extract equations
        equations = await self.equation_parser.extract_equations(text_content, page_num)
        page_data["equations"].extend(equations)
        
        return page_data
    
    def _extract_title(self, text_content: List[Dict]) -> str:
        """Extract document title from text content."""
        if not text_content:
            return ""
        
        # Simple heuristic: title is usually in the first page, larger font
        first_page_text = text_content[0]["content"]
        lines = first_page_text.split('\n')
        
        # Return first non-empty line as title
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                return line
        
        return "Untitled Document"
    
    def _extract_abstract(self, text_content: List[Dict]) -> str:
        """Extract document abstract from text content."""
        if not text_content:
            return ""
        
        full_text = " ".join([item["content"] for item in text_content[:2]])  # First 2 pages
        
        # Find abstract section
        abstract_start = full_text.lower().find("abstract")
        if abstract_start == -1:
            return ""
        
        # Extract text after "abstract"
        abstract_text = full_text[abstract_start + 8:abstract_start + 1000]  # Limit to ~1000 chars
        
        # Clean up the abstract
        abstract_lines = abstract_text.split('\n')
        clean_abstract = " ".join([line.strip() for line in abstract_lines if line.strip()])
        
        return clean_abstract[:500]  # Limit abstract length