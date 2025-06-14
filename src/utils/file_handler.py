import os
import shutil
from pathlib import Path
from typing import List, Optional
import hashlib


class FileHandler:
    """Handle file operations for the research system."""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str, category: str = "papers") -> str:
        """Save uploaded file and return the path."""
        
        category_dir = self.base_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{file_hash}{ext}"
        
        file_path = category_dir / unique_filename
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path)
    
    def list_files(self, category: str = "papers", extension: Optional[str] = None) -> List[str]:
        """List files in a category directory."""
        
        category_dir = self.base_dir / category
        if not category_dir.exists():
            return []
        
        files = []
        for file_path in category_dir.iterdir():
            if file_path.is_file():
                if extension is None or file_path.suffix.lower() == extension.lower():
                    files.append(str(file_path))
        
        return sorted(files)
    
    def cleanup_old_files(self, category: str, max_files: int = 100):
        """Clean up old files to maintain storage limits."""
        
        files = self.list_files(category)
        
        if len(files) > max_files:
            # Sort by modification time and remove oldest
            files_with_time = [(f, os.path.getmtime(f)) for f in files]
            files_with_time.sort(key=lambda x: x[1])
            
            files_to_remove = files_with_time[:-max_files]
            
            for file_path, _ in files_to_remove:
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # File might already be deleted

