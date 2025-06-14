import fitz
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional, List
import logging


class FigureExtractor:
    """Extract and analyze figures from research documents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def extract_figure(self, page, img_info, page_num: int, img_index: int) -> Optional[Dict[str, Any]]:
        """Extract a figure from a page."""
        try:
            # Get image data
            xref = img_info[0]
            pix = fitz.Pixmap(page.parent, xref)
            
            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to numpy array for processing
                img_array = np.array(img)
                
                # Analyze image
                analysis = await self._analyze_figure(img_array)
                
                # Encode image as base64 for storage
                img_b64 = base64.b64encode(img_data).decode()
                
                figure_data = {
                    "page": page_num,
                    "index": img_index,
                    "image_data": img_b64,
                    "format": "png",
                    "analysis": analysis,
                    "type": "figure"
                }
                
                pix = None  # Clean up
                return figure_data
        
        except Exception as e:
            self.logger.error(f"Figure extraction failed: {e}")
        
        return None
    
    async def _analyze_figure(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze figure content and characteristics."""
        height, width = img_array.shape[:2]
        
        # Basic image analysis
        analysis = {
            "dimensions": {"width": width, "height": height},
            "aspect_ratio": width / height,
            "has_text": False,
            "is_chart": False,
            "is_diagram": False,
            "colors": self._analyze_colors(img_array),
            "complexity": self._estimate_complexity(img_array)
        }
        
        # Detect if image contains charts/graphs
        analysis["is_chart"] = self._detect_chart(img_array)
        
        # Detect if image is a diagram
        analysis["is_diagram"] = self._detect_diagram(img_array)
        
        return analysis
    
    def _analyze_colors(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in the image."""
        if len(img_array.shape) == 3:
            # Color image
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
            is_grayscale = False
        else:
            # Grayscale image
            unique_colors = len(np.unique(img_array))
            is_grayscale = True
        
        return {
            "unique_colors": unique_colors,
            "is_grayscale": is_grayscale,
            "is_mostly_white": np.mean(img_array) > 200
        }
    
    def _estimate_complexity(self, img_array: np.ndarray) -> str:
        """Estimate visual complexity of the figure."""
        # Use edge detection to estimate complexity
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        if edge_density < 0.1:
            return "low"
        elif edge_density < 0.3:
            return "medium"
        else:
            return "high"
    
    def _detect_chart(self, img_array: np.ndarray) -> bool:
        """Detect if image contains charts or graphs."""
        # Simple heuristic: look for horizontal and vertical lines
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # If we have both horizontal and vertical lines, likely a chart
        h_score = np.sum(horizontal_lines > 0) / horizontal_lines.size
        v_score = np.sum(vertical_lines > 0) / vertical_lines.size
        
        return h_score > 0.01 and v_score > 0.01
    
    def _detect_diagram(self, img_array: np.ndarray) -> bool:
        """Detect if image is a diagram or flowchart."""
        # Simple heuristic: look for shapes and connections
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count rectangular shapes (common in diagrams)
        rectangular_shapes = 0
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If contour has 4 points, it's likely rectangular
            if len(approx) == 4:
                rectangular_shapes += 1
        
        # If we have multiple rectangular shapes, likely a diagram
        return rectangular_shapes >= 3



