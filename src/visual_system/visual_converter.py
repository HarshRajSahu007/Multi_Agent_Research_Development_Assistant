from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np
from .visualization_generator import VisualizationGenerator


class VisualConverter:
    """Convert between different visualization types."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.viz_generator = VisualizationGenerator(config)
    
    async def convert_visualization(self, source_data: Dict[str, Any], target_type: str) -> Dict[str, Any]:
        """Convert visualization from one type to another."""
        
        try:
            # Extract data from source
            extracted_data = await self._extract_convertible_data(source_data)
            
            # Transform data for target type
            transformed_data = await self._transform_data_for_type(extracted_data, target_type)
            
            # Generate new visualization
            result = await self.viz_generator.generate_from_data(transformed_data, target_type)
            
            return {
                "status": "success",
                "conversion": f"{source_data.get('type', 'unknown')} -> {target_type}",
                "result": result
            }
        
        except Exception as e:
            self.logger.error(f"Visualization conversion failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _extract_convertible_data(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data that can be converted to other formats."""
        
        # This would extract the actual data values from the source visualization
        # For now, we'll create sample data based on the source type
        
        source_type = source_data.get("type", "unknown")
        
        if source_type == "bar_chart":
            return {
                "categories": ["A", "B", "C", "D", "E"],
                "values": [23, 17, 35, 29, 12],
                "title": "Converted Data"
            }
        elif source_type == "line_chart":
            return {
                "x_values": list(range(10)),
                "y_values": [1, 3, 2, 5, 4, 7, 6, 8, 9, 10],
                "title": "Converted Data"
            }
        elif source_type == "pie_chart":
            return {
                "categories": ["Category A", "Category B", "Category C"],
                "values": [40, 35, 25],
                "title": "Converted Data"
            }
        else:
            # Default sample data
            return {
                "categories": ["Item 1", "Item 2", "Item 3"],
                "values": [10, 20, 15],
                "title": "Converted Data"
            }
    
    async def _transform_data_for_type(self, data: Dict[str, Any], target_type: str) -> Dict[str, Any]:
        """Transform data to fit the target visualization type."""
        
        if target_type == "bar_chart":
            # Ensure we have categories and values
            if "categories" not in data and "x_values" in data:
                data["categories"] = [f"Point {i}" for i in range(len(data["x_values"]))]
            if "values" not in data and "y_values" in data:
                data["values"] = data["y_values"]
        
        elif target_type == "line_chart":
            # Ensure we have x and y values
            if "x_values" not in data and "categories" in data:
                data["x_values"] = list(range(len(data["categories"])))
            if "y_values" not in data and "values" in data:
                data["y_values"] = data["values"]
        
        elif target_type == "pie_chart":
            # Ensure we have categories and values, and limit to reasonable number
            if "categories" in data and "values" in data:
                # Limit to top 6 categories for readability
                if len(data["categories"]) > 6:
                    # Sort by values and take top 6
                    combined = list(zip(data["categories"], data["values"]))
                    combined.sort(key=lambda x: x[1], reverse=True)
                    data["categories"] = [item[0] for item in combined[:6]]
                    data["values"] = [item[1] for item in combined[:6]]
        
        elif target_type == "scatter_plot":
            # Generate x,y pairs
            if "values" in data:
                values = data["values"]
                data["x_values"] = list(range(len(values)))
                data["y_values"] = values
        
        elif target_type == "heatmap":
            # Convert to matrix format
            if "values" in data:
                values = data["values"]
                # Create a simple matrix (this is very simplified)
                size = int(np.ceil(np.sqrt(len(values))))
                matrix = np.zeros((size, size))
                for i, val in enumerate(values):
                    row, col = divmod(i, size)
                    if row < size and col < size:
                        matrix[row, col] = val
                data["matrix"] = matrix.tolist()
        
        return data
