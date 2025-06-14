from .base_agent import BaseAgent
from .memory import SharedMemory
from typing import Dict, Any, Optional, List
import ast
import subprocess
import tempfile
import os


class ImplementationAgent(BaseAgent):
    """Agent specialized in implementing algorithms and code from research papers."""
    
    def __init__(self, config: Dict[str, Any], shared_memory: SharedMemory):
        super().__init__(config, "implementation_agent")
        self.shared_memory = shared_memory
    
    async def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process implementation request."""
        self.logger.info(f"Processing implementation request: {request}")
        
        # Get relevant research findings from shared memory
        recent_findings = self.shared_memory.get_recent_findings(3)
        
        # Generate implementation plan
        implementation_plan = await self._create_implementation_plan(request, recent_findings, context)
        
        # Generate code
        code_result = await self._generate_code(request, implementation_plan, context)
        
        # Validate and test code
        validation_result = await self._validate_code(code_result)
        
        # Store implementation in shared memory
        implementation_data = {
            "request": request,
            "plan": implementation_plan,
            "code": code_result,
            "validation": validation_result
        }
        
        self.shared_memory.store_implementation("implementation_agent", implementation_data)
        
        return {
            "status": "completed",
            "agent": "implementation_agent",
            "request": request,
            "implementation": implementation_data
        }
    
    async def _create_implementation_plan(self, request: str, findings: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Create detailed implementation plan."""
        
        findings_text = ""
        if findings:
            findings_text = "\n".join([f"- {finding['content'].get('summary', 'No summary')}" for finding in findings[-3:]])
        
        context_text = ""
        if context:
            context_text = f"Additional context: {context}"
        
        plan_prompt = f"""
        Create a detailed implementation plan for: "{request}"
        
        Related research findings:
        {findings_text}
        
        {context_text}
        
        Please provide:
        1. Algorithm Overview: High-level description of what to implement
        2. Key Components: Main functions/classes needed
        3. Dependencies: Required libraries and tools
        4. Implementation Steps: Step-by-step implementation approach
        5. Testing Strategy: How to validate the implementation
        6. Performance Considerations: Optimization opportunities
        
        Make the plan specific and actionable for a Python implementation.
        """
        
        plan_text = await self._llm_call(plan_prompt, 
                                       "You are a software architecture expert specializing in research implementations.")
        
        return {
            "plan_text": plan_text,
            "components": self._extract_components(plan_text),
            "dependencies": self._extract_dependencies(plan_text),
            "steps": self._extract_steps(plan_text)
        }
    
    async def _generate_code(self, request: str, plan: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate Python code implementation."""
        
        code_prompt = f"""
        Implement the following request in Python: "{request}"
        
        Implementation Plan:
        {plan['plan_text']}
        
        Requirements:
        1. Write clean, well-documented Python code
        2. Include docstrings for all functions and classes
        3. Add type hints where appropriate
        4. Include basic error handling
        5. Make the code modular and reusable
        6. Add example usage at the bottom
        
        Generate complete, runnable Python code.
        """
        
        code_text = await self._llm_call(code_prompt,
                                       "You are an expert Python developer. Generate production-quality code with best practices.")
        
        # Extract actual code from the response
        extracted_code = self._extract_code_blocks(code_text)
        
        return {
            "full_response": code_text,
            "main_code": extracted_code.get("main", ""),
            "additional_files": extracted_code.get("additional", {}),
            "examples": extracted_code.get("examples", "")
        }
    
    async def _validate_code(self, code_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated code for syntax and basic functionality."""
        validation_results = {
            "syntax_valid": False,
            "ast_valid": False,
            "execution_test": False,
            "errors": [],
            "warnings": []
        }
        
        main_code = code_result.get("main_code", "")
        
        if not main_code:
            validation_results["errors"].append("No main code generated")
            return validation_results
        
        # Check syntax
        try:
            compile(main_code, '<string>', 'exec')
            validation_results["syntax_valid"] = True
        except SyntaxError as e:
            validation_results["errors"].append(f"Syntax error: {str(e)}")
        
        # Check AST
        try:
            ast.parse(main_code)
            validation_results["ast_valid"] = True
        except Exception as e:
            validation_results["errors"].append(f"AST parsing error: {str(e)}")
        
        # Basic execution test (in isolated environment)
        if validation_results["syntax_valid"]:
            try:
                execution_result = await self._safe_code_execution(main_code)
                validation_results["execution_test"] = execution_result["success"]
                if not execution_result["success"]:
                    validation_results["errors"].append(f"Execution error: {execution_result['error']}")
            except Exception as e:
                validation_results["errors"].append(f"Execution test failed: {str(e)}")
        
        return validation_results
    
    async def _safe_code_execution(self, code: str) -> Dict[str, Any]:
        """Safely execute code in a temporary environment."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Try to run the code with timeout
            result = subprocess.run(
                ['python', '-c', f'import py_compile; py_compile.compile("{temp_file_path}")'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            success = result.returncode == 0
            error = result.stderr if not success else None
            
            # Clean up
            os.unlink(temp_file_path)
            
            return {
                "success": success,
                "error": error,
                "output": result.stdout
            }
        
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Code execution timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_components(self, plan_text: str) -> List[str]:
        """Extract key components from implementation plan."""
        components = []
        lines = plan_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'components' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                components.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return components
    
    def _extract_dependencies(self, plan_text: str) -> List[str]:
        """Extract dependencies from implementation plan."""
        dependencies = []
        lines = plan_text.split('\n')
        
        for line in lines:
            if 'import' in line or 'pip install' in line:
                dependencies.append(line.strip())
        
        # Common research dependencies
        common_deps = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'torch', 'tensorflow']
        for dep in common_deps:
            if dep in plan_text.lower():
                dependencies.append(dep)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_steps(self, plan_text: str) -> List[str]:
        """Extract implementation steps from plan."""
        steps = []
        lines = plan_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'steps' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                steps.append(line.strip())
            elif capturing and ('testing' in line.lower() or 'performance' in line.lower()):
                break
        
        return steps
    
    def _extract_code_blocks(self, text: str) -> Dict[str, Any]:
        """Extract code blocks from LLM response."""
        import re
        
        # Pattern to match code blocks
        code_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            main_code = matches[0].strip()
            additional = {}
            
            # If multiple code blocks, treat others as additional files
            for i, match in enumerate(matches[1:], 1):
                additional[f"file_{i}.py"] = match.strip()
            
            return {
                "main": main_code,
                "additional": additional,
                "examples": matches[-1] if len(matches) > 1 else ""
            }
        else:
            # If no code blocks found, assume entire response is code
            return {
                "main": text.strip(),
                "additional": {},
                "examples": ""
            }
