import re
from typing import List, Dict, Any
import logging


class EquationParser:
    """Parse and extract mathematical equations from text."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def extract_equations(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract mathematical equations from text."""
        equations = []
        
        # LaTeX equation patterns
        latex_patterns = [
            r'\$\$(.+?)\$\$',  # Display math
            r'\$(.+?)\$',      # Inline math
            r'\\begin\{equation\}(.+?)\\end\{equation\}',  # Equation environment
            r'\\begin\{align\}(.+?)\\end\{align\}',        # Align environment
        ]
        
        for pattern in latex_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                equation_text = match.group(1).strip()
                if equation_text:
                    equation_data = {
                        "page": page_num,
                        "content": equation_text,
                        "type": "latex",
                        "position": match.span(),
                        "analysis": self._analyze_equation(equation_text)
                    }
                    equations.append(equation_data)
        
        # Simple mathematical expressions
        math_patterns = [
            r'([a-zA-Z]\s*=\s*[^=\n]+)',  # Variable assignments
            r'(\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*\s*=\s*\d+\.?\d*)',  # Simple arithmetic
        ]
        
        for pattern in math_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                equation_text = match.group(1).strip()
                if len(equation_text) > 3:  # Filter very short matches
                    equation_data = {
                        "page": page_num,
                        "content": equation_text,
                        "type": "arithmetic",
                        "position": match.span(),
                        "analysis": self._analyze_equation(equation_text)
                    }
                    equations.append(equation_data)
        
        return equations
    
    def _analyze_equation(self, equation_text: str) -> Dict[str, Any]:
        """Analyze mathematical equation content."""
        analysis = {
            "length": len(equation_text),
            "has_variables": bool(re.search(r'[a-zA-Z]', equation_text)),
            "has_numbers": bool(re.search(r'\d', equation_text)),
            "has_operators": bool(re.search(r'[+\-*/=<>]', equation_text)),
            "has_functions": bool(re.search(r'(sin|cos|tan|log|exp|sqrt)', equation_text)),
            "has_greek_letters": bool(re.search(r'(alpha|beta|gamma|delta|theta|lambda|mu|sigma|pi)', equation_text)),
            "complexity": "simple"
        }
        
        # Estimate complexity
        complexity_score = 0
        if analysis["has_variables"]: complexity_score += 1
        if analysis["has_functions"]: complexity_score += 2
        if analysis["has_greek_letters"]: complexity_score += 1
        if len(equation_text) > 20: complexity_score += 1
        
        if complexity_score >= 3:
            analysis["complexity"] = "complex"
        elif complexity_score >= 1:
            analysis["complexity"] = "medium"
        
        return analysis
