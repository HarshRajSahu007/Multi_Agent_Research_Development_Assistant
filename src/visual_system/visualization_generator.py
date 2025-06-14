import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import base64
import io


class VisualizationGenerator:
    """Generate new visualizations based on research findings."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set style preferences
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    async def generate_from_data(self, data: Dict[str, Any], viz_type: str = "auto") -> Dict[str, Any]:
        """Generate visualization from structured data."""
        
        try:
            if viz_type == "auto":
                viz_type = self._determine_best_viz_type(data)
            
            # Generate the visualization
            if viz_type == "bar_chart":
                result = await self._create_bar_chart(data)
            elif viz_type == "line_chart":
                result = await self._create_line_chart(data)
            elif viz_type == "scatter_plot":
                result = await self._create_scatter_plot(data)
            elif viz_type == "pie_chart":
                result = await self._create_pie_chart(data)
            elif viz_type == "heatmap":
                result = await self._create_heatmap(data)
            else:
                result = await self._create_default_visualization(data)
            
            return {
                "status": "success",
                "visualization": result,
                "type": viz_type
            }
        
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _determine_best_viz_type(self, data: Dict[str, Any]) -> str:
        """Determine the best visualization type for the data."""
        
        # Simple heuristics for chart type selection
        if "categories" in data and "values" in data:
            if len(data["categories"]) <= 6:
                return "pie_chart"
            else:
                return "bar_chart"
        elif "x_values" in data and "y_values" in data:
            if len(data["x_values"]) > 20:
                return "line_chart"
            else:
                return "scatter_plot"
        elif "matrix" in data:
            return "heatmap"
        else:
            return "bar_chart"  # Default fallback
    
    async def _create_bar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a bar chart visualization."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = data.get("categories", [f"Cat {i}" for i in range(len(data.get("values", [])))])
        values = data.get("values", [1, 2, 3, 4, 5])
        
        bars = ax.bar(categories, values)
        
        # Customize appearance
        ax.set_title(data.get("title", "Bar Chart"))
        ax.set_xlabel(data.get("x_label", "Categories"))
        ax.set_ylabel(data.get("y_label", "Values"))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_data": img_b64,
            "chart_type": "bar_chart",
            "description": f"Bar chart showing {len(categories)} categories"
        }
    
    async def _create_line_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a line chart visualization."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_values = data.get("x_values", list(range(len(data.get("y_values", [])))))
        y_values = data.get("y_values", [1, 2, 3, 4, 5])
        
        ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=6)
        
        # Customize appearance
        ax.set_title(data.get("title", "Line Chart"))
        ax.set_xlabel(data.get("x_label", "X Values"))
        ax.set_ylabel(data.get("y_label", "Y Values"))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_data": img_b64,
            "chart_type": "line_chart",
            "description": f"Line chart with {len(x_values)} data points"
        }
    
    async def _create_scatter_plot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a scatter plot visualization."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x_values = data.get("x_values", np.random.randn(50))
        y_values = data.get("y_values", np.random.randn(50))
        
        scatter = ax.scatter(x_values, y_values, alpha=0.6, s=60)
        
        # Customize appearance
        ax.set_title(data.get("title", "Scatter Plot"))
        ax.set_xlabel(data.get("x_label", "X Values"))
        ax.set_ylabel(data.get("y_label", "Y Values"))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_data": img_b64,
            "chart_type": "scatter_plot",
            "description": f"Scatter plot with {len(x_values)} points"
        }
    
    async def _create_pie_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a pie chart visualization."""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        categories = data.get("categories", [f"Cat {i}" for i in range(len(data.get("values", [])))])
        values = data.get("values", [1, 2, 3, 4])
        
        wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
        
        # Customize appearance
        ax.set_title(data.get("title", "Pie Chart"))
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_data": img_b64,
            "chart_type": "pie_chart",
            "description": f"Pie chart with {len(categories)} segments"
        }
    
    async def _create_heatmap(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a heatmap visualization."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate sample matrix if not provided
        if "matrix" in data:
            matrix = np.array(data["matrix"])
        else:
            matrix = np.random.randn(10, 10)
        
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im)
        
        # Customize appearance
        ax.set_title(data.get("title", "Heatmap"))
        ax.set_xlabel(data.get("x_label", "X Axis"))
        ax.set_ylabel(data.get("y_label", "Y Axis"))
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_data": img_b64,
            "chart_type": "heatmap",
            "description": f"Heatmap with {matrix.shape[0]}x{matrix.shape[1]} cells"
        }
    
    async def _create_default_visualization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a default visualization when type is unclear."""
        
        # Create a simple bar chart as default
        return await self._create_bar_chart({
            "categories": ["A", "B", "C", "D"],
            "values": [1, 3, 2, 4],
            "title": "Default Visualization",
            "x_label": "Categories",
            "y_label": "Values"
        })
