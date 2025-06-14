import fitz
import pandas as pd
from typing import List, Dict, Any
import logging
import re


class TableExtractor:
    """Extract and parse tables from research documents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def extract_tables(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables from a page."""
        tables = []
        
        try:
            # Try to find tables using PyMuPDF's table detection
            page_tables = page.find_tables()
            
            for table_index, table in enumerate(page_tables):
                table_data = self._process_table(table, page_num, table_index)
                if table_data:
                    tables.append(table_data)
        
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
        
        return tables
    
    def _process_table(self, table, page_num: int, table_index: int) -> Dict[str, Any]:
        """Process an individual table."""
        try:
            # Extract table data
            table_data = table.extract()
            
            if not table_data:
                return None
            
            # Convert to pandas DataFrame for easier processing
            df = pd.DataFrame(table_data[1:], columns=table_data[0])  # First row as headers
            
            # Clean the data
            df = self._clean_table_data(df)
            
            # Analyze table structure
            analysis = self._analyze_table(df)
            
            return {
                "page": page_num,
                "index": table_index,
                "data": df.to_dict('records'),
                "headers": list(df.columns),
                "shape": df.shape,
                "analysis": analysis,
                "type": "table"
            }
        
        except Exception as e:
            self.logger.error(f"Table processing failed: {e}")
            return None
    
    def _clean_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean table data by removing empty rows/columns and fixing formatting."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Try to convert numeric columns
        for col in df.columns:
            # Check if column contains mostly numbers
            numeric_count = sum(1 for val in df[col] if self._is_numeric(str(val)))
            if numeric_count > len(df) * 0.7:  # 70% numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string represents a number."""
        try:
            float(value.replace(',', '').replace('%', '').replace('$', ''))
            return True
        except (ValueError, AttributeError):
            return False
    
    def _analyze_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze table structure and content."""
        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "has_numeric_data": any(df.dtypes == 'float64'),
            "has_headers": True,  # Assumed since we use first row as headers
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "completeness": (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
        
        return analysis