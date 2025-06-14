import easyocr
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
import logging


class OCREngine:
    """Specialized OCR engine for mathematical notation and technical diagrams."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])
    
    async def extract_text_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text from image using OCR."""
        try:
            # Use EasyOCR to extract text
            results = self.reader.readtext(image)
            
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    extracted_text.append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence,
                        "type": self._classify_text_type(text)
                    })
            
            return extracted_text
        
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return []
    
    def _classify_text_type(self, text: str) -> str:
        """Classify the type of extracted text."""
        # Simple heuristics for text classification
        if any(char in text for char in ['∫', '∑', '∂', 'α', 'β', 'γ', 'θ', 'λ', 'μ', 'σ']):
            return "mathematical"
        elif any(char in text for char in ['=', '+', '-', '*', '/', '^', '²', '³']):
            return "equation"
        elif text.isupper() and len(text) < 10:
            return "label"
        else:
            return "general"