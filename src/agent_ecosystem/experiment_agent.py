from .base_agent import BaseAgent
from .memory import SharedMemory
from typing import Dict, Any, Optional, List
import asyncio
import json
import numpy as np
from datetime import datetime


class ExperimentAgent(BaseAgent):
    """Agent specialized in designing and executing experiments."""
    
    def __init__(self, config: Dict[str, Any], shared_memory: SharedMemory):
        super().__init__(config, "experiment_agent")
        self.shared_memory = shared_memory
    
    async def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process experiment design and execution request."""
        self.logger.info(f"Processing experiment request: {request}")
        
        # Get relevant context from shared memory
        relevant_findings = self.shared_memory.get_recent_findings(3)
        recent_implementations = self.shared_memory.memory["implementations"][-2:]
        
        # Design experiment
        experiment_design = await self._design_experiment(request, relevant_findings, recent_implementations, context)
        
        # Execute experiment (simulation)
        execution_results = await self._execute_experiment(experiment_design)
        
        # Analyze results
        analysis = await self._analyze_results(experiment_design, execution_results)
        
        # Store experiment in shared memory
        experiment_data = {
            "request": request,
            "design": experiment_design,
            "results": execution_results,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        self.shared_memory.store_experiment("experiment_agent", experiment_data)
        
        return {
            "status": "completed", 
            "agent": "experiment_agent",
            "request": request,
            "experiment": experiment_data
        }
    
    async def _design_experiment(self, request: str, findings: List[Dict], implementations: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Design experiment based on request and available context."""
        
        findings_summary = ""
        if findings:
            findings_summary = "\n".join([f"- {f['content'].get('summary', 'No summary')[:100]}" for f in findings])
        
        impl_summary = ""
        if implementations:
            impl_summary = "\n".join([f"- {impl['content'].get('request', 'No description')[:100]}" for impl in implementations])
        
        design_prompt = f"""
        Design a comprehensive experiment for: "{request}"
        
        Available research findings:
        {findings_summary}
        
        Available implementations:
        {impl_summary}
        
        Please design an experiment with:
        1. Hypothesis: Clear, testable hypothesis
        2. Variables: Independent and dependent variables
        3. Methodology: Experimental approach and procedures
        4. Metrics: Success metrics and evaluation criteria
        5. Controls: Control conditions and baselines
        6. Data Requirements: What data is needed
        7. Expected Outcomes: Predicted results
        8. Validation Strategy: How to validate results
        
        Make the experiment practical and executable with available resources.
        """
        
        design_text = await self._llm_call(design_prompt,
                                         "You are an experimental design expert. Create rigorous, well-structured experiments.")
        
        return {
            "design_text": design_text,
            "hypothesis": self._extract_hypothesis(design_text),
            "variables": self._extract_variables(design_text),
            "methodology": self._extract_methodology(design_text),
            "metrics": self._extract_metrics(design_text),
            "data_requirements": self._extract_data_requirements(design_text)
        }
    
    async def _execute_experiment(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the experiment (simulation for demonstration)."""
        self.logger.info("Executing experiment simulation")
        
        # Simulate experiment execution
        # In a real implementation, this would execute actual experiments
        
        # Generate simulated data based on experiment design
        simulated_data = self._generate_simulated_data(design)
        
        # Simulate running the experiment
        await asyncio.sleep(1)  # Simulate processing time
        
        # Generate results
        results = {
            "execution_status": "completed",
            "data_collected": simulated_data,
            "metrics_calculated": self._calculate_simulated_metrics(simulated_data, design),
            "observations": self._generate_observations(design),
            "raw_data_size": len(simulated_data.get("samples", [])),
            "execution_time": "simulated"
        }
        
        return results
    
    async def _analyze_results(self, design: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results."""
        
        analysis_prompt = f"""
        Analyze the following experiment results:
        
        Experiment Design:
        Hypothesis: {design.get('hypothesis', 'Not specified')}
        Methodology: {design.get('methodology', 'Not specified')}
        
        Results Summary:
        - Data points collected: {results.get('raw_data_size', 0)}
        - Metrics: {json.dumps(results.get('metrics_calculated', {}), indent=2)}
        - Observations: {results.get('observations', [])}
        
        Please provide:
        1. Hypothesis Validation: Whether the hypothesis was supported
        2. Key Findings: Main discoveries from the experiment
        3. Statistical Significance: Assessment of result reliability
        4. Limitations: Experiment limitations and potential biases
        5. Implications: What these results mean for the research
        6. Next Steps: Recommended follow-up experiments
        
        Provide a thorough scientific analysis.
        """
        
        analysis_text = await self._llm_call(analysis_prompt,
                                           "You are a research analyst expert in experimental result interpretation.")
        
        return {
            "analysis_text": analysis_text,
            "hypothesis_supported": self._assess_hypothesis_support(analysis_text),
            "key_findings": self._extract_key_findings(analysis_text),
            "statistical_significance": self._assess_significance(results),
            "limitations": self._extract_limitations(analysis_text),
            "next_steps": self._extract_next_steps(analysis_text)
        }
    
    def _generate_simulated_data(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated experimental data."""
        # Simple simulation based on common experimental patterns
        
        np.random.seed(42)  # For reproducible results
        
        # Generate sample data
        n_samples = 100
        
        # Simulate different types of experimental data
        data = {
            "samples": [],
            "conditions": ["control", "treatment_a", "treatment_b"],
            "measurements": {}
        }
        
        for condition in data["conditions"]:
            # Generate measurements for each condition
            if "control" in condition:
                measurements = np.random.normal(50, 10, n_samples)
            elif "treatment_a" in condition:
                measurements = np.random.normal(55, 12, n_samples)  # Slight improvement
            else:
                measurements = np.random.normal(60, 8, n_samples)   # Better improvement
            
            data["measurements"][condition] = measurements.tolist()
            
            # Create sample records
            for i, measurement in enumerate(measurements):
                data["samples"].append({
                    "id": f"{condition}_{i}",
                    "condition": condition,
                    "measurement": float(measurement),
                    "timestamp": i
                })
        
        return data
    
    def _calculate_simulated_metrics(self, data: Dict[str, Any], design: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics from simulated data."""
        metrics = {}
        
        measurements = data.get("measurements", {})
        
        for condition, values in measurements.items():
            metrics[condition] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "count": len(values)
            }
        
        # Calculate comparative metrics
        if len(measurements) > 1:
            conditions = list(measurements.keys())
            control = measurements.get("control", measurements[conditions[0]])
            
            for condition, values in measurements.items():
                if condition != "control" and len(control) > 0:
                    improvement = (np.mean(values) - np.mean(control)) / np.mean(control) * 100
                    metrics[f"{condition}_vs_control"] = {
                        "improvement_percent": float(improvement),
                        "effect_size": float((np.mean(values) - np.mean(control)) / np.sqrt((np.var(values) + np.var(control)) / 2))
                    }
        
        return metrics
    
    def _generate_observations(self, design: Dict[str, Any]) -> List[str]:
        """Generate simulated observations."""
        observations = [
            "Experiment executed successfully with all planned conditions",
            "Data collection completed within expected parameters",
            "No significant anomalies detected during execution",
            "All measurement instruments performed within tolerance",
            "Sample sizes adequate for statistical analysis"
        ]
        
        return observations
    
    def _extract_hypothesis(self, design_text: str) -> str:
        """Extract hypothesis from design text."""
        lines = design_text.split('\n')
        for i, line in enumerate(lines):
            if 'hypothesis' in line.lower():
                # Return next line or current line after colon
                if ':' in line:
                    return line.split(':', 1)[1].strip()
                elif i + 1 < len(lines):
                    return lines[i + 1].strip()
        return "Hypothesis not clearly specified"
    
    def _extract_variables(self, design_text: str) -> Dict[str, List[str]]:
        """Extract variables from design text."""
        variables = {"independent": [], "dependent": []}
        
        lines = design_text.split('\n')
        current_type = None
        
        for line in lines:
            line_lower = line.lower()
            if 'independent' in line_lower:
                current_type = "independent"
            elif 'dependent' in line_lower:
                current_type = "dependent"
            elif current_type and line.startswith(('1.', '2.', '3.', '-', '•')):
                variables[current_type].append(line.strip())
        
        return variables
    
    def _extract_methodology(self, design_text: str) -> str:
        """Extract methodology from design text."""
        lines = design_text.split('\n')
        for i, line in enumerate(lines):
            if 'methodology' in line.lower():
                # Return next few lines
                methodology_lines = lines[i+1:i+4]
                return ' '.join([line.strip() for line in methodology_lines if line.strip()])
        return "Methodology not specified"
    
    def _extract_metrics(self, design_text: str) -> List[str]:
        """Extract metrics from design text."""
        metrics = []
        lines = design_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'metrics' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                metrics.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return metrics
    
    def _extract_data_requirements(self, design_text: str) -> List[str]:
        """Extract data requirements from design text."""
        requirements = []
        lines = design_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'data' in line.lower() and 'requirement' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                requirements.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return requirements
    
    def _assess_hypothesis_support(self, analysis_text: str) -> bool:
        """Assess if hypothesis was supported based on analysis."""
        analysis_lower = analysis_text.lower()
        support_indicators = ['supported', 'confirmed', 'validated', 'proven']
        reject_indicators = ['rejected', 'not supported', 'disproven', 'failed']
        
        support_count = sum(1 for indicator in support_indicators if indicator in analysis_lower)
        reject_count = sum(1 for indicator in reject_indicators if indicator in analysis_lower)
        
        return support_count > reject_count
    
    def _extract_key_findings(self, analysis_text: str) -> List[str]:
        """Extract key findings from analysis."""
        findings = []
        lines = analysis_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'key findings' in line.lower() or 'findings' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                findings.append(line.strip())
            elif capturing and ('limitations' in line.lower() or 'implications' in line.lower()):
                break
        
        return findings
    
    def _assess_significance(self, results: Dict[str, Any]) -> str:
        """Assess statistical significance of results."""
        metrics = results.get("metrics_calculated", {})
        
        # Simple heuristic based on effect sizes
        effect_sizes = []
        for key, value in metrics.items():
            if "effect_size" in str(value):
                if isinstance(value, dict) and "effect_size" in value:
                    effect_sizes.append(abs(value["effect_size"]))
        
        if not effect_sizes:
            return "Cannot determine - insufficient statistical data"
        
        max_effect = max(effect_sizes)
        if max_effect > 0.8:
            return "Large effect size - likely significant"
        elif max_effect > 0.5:
            return "Medium effect size - potentially significant"
        elif max_effect > 0.2:
            return "Small effect size - may not be significant"
        else:
            return "Very small effect size - likely not significant"
    
    def _extract_limitations(self, analysis_text: str) -> List[str]:
        """Extract limitations from analysis."""
        limitations = []
        lines = analysis_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'limitations' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                limitations.append(line.strip())
            elif capturing and ('implications' in line.lower() or 'next steps' in line.lower()):
                break
        
        return limitations
    
    def _extract_next_steps(self, analysis_text: str) -> List[str]:
        """Extract next steps from analysis."""
        next_steps = []
        lines = analysis_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'next steps' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                next_steps.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return next_steps