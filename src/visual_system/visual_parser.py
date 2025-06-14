import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging
import base64
import io
import json


class VisualParser:
    """Parse and understand scientific visualizations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def parse_visualization(self, image_data: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parse a scientific visualization and extract information."""
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(image)
            
            # Analyze the visualization
            analysis = await self._comprehensive_visual_analysis(img_array, context)
            
            return {
                "status": "success",
                "analysis": analysis,
                "image_info": {
                    "dimensions": image.size,
                    "format": image.format,
                    "mode": image.mode
                }
            }
        
        except Exception as e:
            self.logger.error(f"Visual parsing failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _comprehensive_visual_analysis(self, img_array: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis of the visualization."""
        
        analysis = {
            "chart_type": self._identify_chart_type(img_array),
            "data_extraction": await self._extract_data_points(img_array),
            "text_elements": await self._extract_text_elements(img_array),
            "color_analysis": self._analyze_colors(img_array),
            "layout_analysis": self._analyze_layout(img_array),
            "insights": await self._generate_insights(img_array, context)
        }
        
        return analysis
    
    def _identify_chart_type(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Identify the type of chart or visualization."""
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Detect lines and shapes
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Calculate line densities
        h_density = np.sum(horizontal_lines > 0) / horizontal_lines.size
        v_density = np.sum(vertical_lines > 0) / vertical_lines.size
        
        # Detect circles (for pie charts)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        has_circles = circles is not None
        
        # Classify chart type based on features
        chart_type = "unknown"
        confidence = 0.0
        features = {
            "has_horizontal_lines": h_density > 0.01,
            "has_vertical_lines": v_density > 0.01,
            "has_circles": has_circles,
            "line_density": h_density + v_density
        }
        
        if has_circles:
            chart_type = "pie_chart"
            confidence = 0.8
        elif h_density > 0.02 and v_density > 0.02:
            chart_type = "bar_chart_or_line_chart"
            confidence = 0.7
        elif h_density > 0.01 or v_density > 0.01:
            chart_type = "chart_with_axes"
            confidence = 0.6
        else:
            chart_type = "diagram_or_image"
            confidence = 0.5
        
        return {
            "type": chart_type,
            "confidence": confidence,
            "features": features
        }
    
    async def _extract_data_points(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Extract data points from the visualization."""
        
        # This is a simplified implementation
        # In a real system, you'd use more sophisticated computer vision techniques
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Find contours that might represent data points
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        data_points = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) > 10:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                data_points.append({
                    "x": int(center_x),
                    "y": int(center_y),
                    "area": float(cv2.contourArea(contour)),
                    "bbox": [int(x), int(y), int(w), int(h)]
                })
        
        # Sort by y-coordinate (top to bottom)
        data_points.sort(key=lambda p: p["y"])
        
        return {
            "points": data_points[:20],  # Limit to 20 points
            "total_points": len(data_points),
            "extraction_method": "contour_detection"
        }
    
    async def _extract_text_elements(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Extract text elements from the visualization."""
        
        # This would typically use OCR
        # For now, we'll return a placeholder structure
        
        return {
            "title": "Chart Title (extracted via OCR)",
            "axis_labels": {
                "x_axis": "X-axis Label",
                "y_axis": "Y-axis Label"
            },
            "legend": ["Series 1", "Series 2"],
            "annotations": [],
            "extraction_method": "ocr_placeholder"
        }
    
    def _analyze_colors(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color usage in the visualization."""
        
        if len(img_array.shape) == 2:
            # Grayscale image
            return {
                "is_grayscale": True,
                "dominant_colors": ["gray"],
                "color_count": 1
            }
        
        # Reshape image to be a list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Find unique colors (simplified)
        unique_colors = np.unique(pixels, axis=0)
        
        # Calculate color statistics
        dominant_color = np.mean(pixels, axis=0).astype(int)
        
        return {
            "is_grayscale": False,
            "unique_color_count": len(unique_colors),
            "dominant_color": dominant_color.tolist(),
            "color_diversity": len(unique_colors) / len(pixels) if len(pixels) > 0 else 0
        }
    
    def _analyze_layout(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze the layout and structure of the visualization."""
        
        height, width = img_array.shape[:2]
        
        # Divide image into regions
        regions = {
            "top": img_array[:height//4, :],
            "bottom": img_array[3*height//4:, :],
            "left": img_array[:, :width//4],
            "right": img_array[:, 3*width//4:],
            "center": img_array[height//4:3*height//4, width//4:3*width//4]
        }
        
        # Analyze content density in each region
        layout_analysis = {}
        for region_name, region_data in regions.items():
            # Calculate edge density as a proxy for content
            if len(region_data.shape) == 3:
                region_gray = cv2.cvtColor(region_data, cv2.COLOR_RGB2GRAY)
            else:
                region_gray = region_data
            
            edges = cv2.Canny(region_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            layout_analysis[region_name] = {
                "edge_density": float(edge_density),
                "mean_intensity": float(np.mean(region_gray)),
                "content_level": "high" if edge_density > 0.1 else "medium" if edge_density > 0.05 else "low"
            }
        
        return layout_analysis
    
    async def _generate_insights(self, img_array: np.ndarray, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate insights about the visualization."""
        
        insights = []
        
        # Basic insights based on image properties
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > 1.5:
            insights.append("Wide format visualization, likely optimized for horizontal data display")
        elif aspect_ratio < 0.7:
            insights.append("Tall format visualization, possibly showing hierarchical or vertical data")
        
        # Color insights
        if len(img_array.shape) == 2:
            insights.append("Grayscale visualization, focusing on data rather than aesthetic appeal")
        else:
            color_variance = np.var(img_array)
            if color_variance > 1000:
                insights.append("High color variance suggests multiple data series or categories")
            else:
                insights.append("Low color variance suggests simple or monochromatic design")
        
        # Edge density insights
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.15:
            insights.append("High edge density indicates complex visualization with detailed elements")
        elif edge_density < 0.05:
            insights.append("Low edge density suggests simple, clean visualization design")
        
        return insights
