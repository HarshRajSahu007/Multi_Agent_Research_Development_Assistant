from .base_agent import BaseAgent
from .memory import SharedMemory
from typing import Dict, Any, Optional, List


class CritiqueAgent(BaseAgent):
    """Agent specialized in evaluating research claims and identifying weaknesses."""
    
    def __init__(self, config: Dict[str, Any], shared_memory: SharedMemory):
        super().__init__(config, "critique_agent")
        self.shared_memory = shared_memory
    
    async def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process critique and evaluation request."""
        self.logger.info(f"Processing critique request: {request}")
        
        # Get relevant content from shared memory
        recent_findings = self.shared_memory.get_recent_findings(3)
        recent_experiments = self.shared_memory.memory["experiments"][-2:]
        recent_implementations = self.shared_memory.memory["implementations"][-2:]
        
        # Perform comprehensive critique
        critique_result = await self._perform_critique(request, recent_findings, recent_experiments, recent_implementations, context)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(critique_result)
        
        # Store critique in shared memory
        critique_data = {
            "request": request,
            "critique": critique_result,
            "recommendations": recommendations
        }
        
        self.shared_memory.store_critique("critique_agent", critique_data)
        
        return {
            "status": "completed",
            "agent": "critique_agent", 
            "request": request,
            "critique": critique_data
        }
    
    async def _perform_critique(self, request: str, findings: List[Dict], experiments: List[Dict], implementations: List[Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform comprehensive critique of research work."""
        
        # Prepare content for critique
        findings_summary = self._summarize_findings(findings)
        experiments_summary = self._summarize_experiments(experiments)
        implementations_summary = self._summarize_implementations(implementations)
        
        critique_prompt = f"""
        Perform a comprehensive critique of the following research work related to: "{request}"
        
        Research Findings:
        {findings_summary}
        
        Experiments Conducted:
        {experiments_summary}
        
        Implementations Created:
        {implementations_summary}
        
        Please provide a detailed critique covering:
        1. Methodological Rigor: Assessment of research methodology quality
        2. Evidence Quality: Evaluation of evidence strength and reliability
        3. Logical Consistency: Check for logical gaps or contradictions
        4. Completeness: Identification of missing elements or gaps
        5. Reproducibility: Assessment of work reproducibility
        6. Novelty: Evaluation of contribution novelty and significance
        7. Limitations: Identification of key limitations and constraints
        8. Potential Biases: Detection of possible biases or assumptions
        9. Alternative Explanations: Consideration of alternative interpretations
        10. Improvement Suggestions: Specific recommendations for enhancement
        
        Be thorough but constructive in your critique.
        """
        
        critique_text = await self._llm_call(critique_prompt,
                                           "You are a rigorous research critic with expertise in scientific methodology. Provide constructive, detailed analysis.")
        
        return {
            "critique_text": critique_text,
            "methodological_score": self._assess_methodology(critique_text),
            "evidence_score": self._assess_evidence(critique_text),
            "completeness_score": self._assess_completeness(critique_text),
            "reproducibility_score": self._assess_reproducibility(critique_text),
            "identified_issues": self._extract_issues(critique_text),
            "strengths": self._extract_strengths(critique_text),
            "weaknesses": self._extract_weaknesses(critique_text)
        }
    
    async def _generate_recommendations(self, critique_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations based on critique."""
        
        recommendations_prompt = f"""
        Based on the following critique, generate specific, actionable recommendations:
        
        Critique Summary:
        {critique_result['critique_text']}
        
        Identified Issues:
        {critique_result.get('identified_issues', [])}
        
        Please provide:
        1. Priority Improvements: Most critical issues to address first
        2. Methodological Improvements: Specific methodology enhancements
        3. Additional Experiments: Suggested follow-up experiments
        4. Implementation Fixes: Code or implementation improvements needed
        5. Documentation Improvements: Areas needing better documentation
        6. Validation Steps: Additional validation measures to implement
        
        Make recommendations specific, actionable, and prioritized.
        """
        
        recommendations_text = await self._llm_call(recommendations_prompt,
                                                  "You are a research advisor providing actionable guidance for improvement.")
        
        return {
            "recommendations_text": recommendations_text,
            "priority_improvements": self._extract_priority_improvements(recommendations_text),
            "methodological_improvements": self._extract_methodological_improvements(recommendations_text),
            "additional_experiments": self._extract_additional_experiments(recommendations_text),
            "implementation_fixes": self._extract_implementation_fixes(recommendations_text),
            "validation_steps": self._extract_validation_steps(recommendations_text)
        }
    
    def _summarize_findings(self, findings: List[Dict]) -> str:
        """Summarize research findings for critique."""
        if not findings:
            return "No recent findings available."
        
        summary = "Recent Research Findings:\n"
        for i, finding in enumerate(findings, 1):
            content = finding.get('content', {})
            summary += f"{i}. {content.get('summary', 'No summary available')}\n"
            if content.get('insights'):
                summary += f"   Key insights: {', '.join(content['insights'][:2])}\n"
        
        return summary
    
    def _summarize_experiments(self, experiments: List[Dict]) -> str:
        """Summarize experiments for critique."""
        if not experiments:
            return "No recent experiments available."
        
        summary = "Recent Experiments:\n"
        for i, exp in enumerate(experiments, 1):
            content = exp.get('content', {})
            summary += f"{i}. {content.get('request', 'No description')}\n"
            if content.get('design', {}).get('hypothesis'):
                summary += f"   Hypothesis: {content['design']['hypothesis']}\n"
            if content.get('analysis', {}).get('hypothesis_supported') is not None:
                supported = content['analysis']['hypothesis_supported']
                summary += f"   Result: {'Hypothesis supported' if supported else 'Hypothesis not supported'}\n"
        
        return summary
    
    def _summarize_implementations(self, implementations: List[Dict]) -> str:
        """Summarize implementations for critique."""
        if not implementations:
            return "No recent implementations available."
        
        summary = "Recent Implementations:\n"
        for i, impl in enumerate(implementations, 1):
            content = impl.get('content', {})
            summary += f"{i}. {content.get('request', 'No description')}\n"
            if content.get('validation', {}).get('syntax_valid') is not None:
                validation = content['validation']
                summary += f"   Validation: {'Passed' if validation.get('syntax_valid') else 'Failed'}\n"
        
        return summary
    
    def _assess_methodology(self, critique_text: str) -> int:
        """Assess methodological rigor (1-10 scale)."""
        critique_lower = critique_text.lower()
        
        positive_indicators = ['rigorous', 'systematic', 'well-designed', 'appropriate', 'sound']
        negative_indicators = ['flawed', 'inadequate', 'poor', 'weak', 'problematic']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in critique_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in critique_lower)
        
        # Simple scoring heuristic
        if positive_count > negative_count * 2:
            return 8
        elif positive_count > negative_count:
            return 6
        elif negative_count > positive_count:
            return 4
        else:
            return 5
    
    def _assess_evidence(self, critique_text: str) -> int:
        """Assess evidence quality (1-10 scale)."""
        critique_lower = critique_text.lower()
        
        strong_evidence = ['strong evidence', 'compelling', 'robust', 'comprehensive']
        weak_evidence = ['weak evidence', 'insufficient', 'limited', 'anecdotal']
        
        strong_count = sum(1 for indicator in strong_evidence if indicator in critique_lower)
        weak_count = sum(1 for indicator in weak_evidence if indicator in critique_lower)
        
        if strong_count > weak_count:
            return 7
        elif weak_count > strong_count:
            return 4
        else:
            return 5
    
    def _assess_completeness(self, critique_text: str) -> int:
        """Assess completeness (1-10 scale)."""
        critique_lower = critique_text.lower()
        
        complete_indicators = ['comprehensive', 'complete', 'thorough', 'detailed']
        incomplete_indicators = ['missing', 'incomplete', 'gaps', 'lacking']
        
        complete_count = sum(1 for indicator in complete_indicators if indicator in critique_lower)
        incomplete_count = sum(1 for indicator in incomplete_indicators if indicator in critique_lower)
        
        if complete_count > incomplete_count:
            return 7
        elif incomplete_count > complete_count:
            return 4
        else:
            return 5
    
    def _assess_reproducibility(self, critique_text: str) -> int:
        """Assess reproducibility (1-10 scale)."""
        critique_lower = critique_text.lower()
        
        reproducible_indicators = ['reproducible', 'replicable', 'documented', 'clear']
        non_reproducible_indicators = ['not reproducible', 'unclear', 'undocumented', 'ambiguous']
        
        repro_count = sum(1 for indicator in reproducible_indicators if indicator in critique_lower)
        non_repro_count = sum(1 for indicator in non_reproducible_indicators if indicator in critique_lower)
        
        if repro_count > non_repro_count:
            return 7
        elif non_repro_count > repro_count:
            return 4
        else:
            return 5
    
    def _extract_issues(self, critique_text: str) -> List[str]:
        """Extract identified issues from critique."""
        issues = []
        lines = critique_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['issue', 'problem', 'weakness', 'limitation', 'flaw']):
                issues.append(line.strip())
        
        return issues[:5]  # Limit to top 5 issues
    
    def _extract_strengths(self, critique_text: str) -> List[str]:
        """Extract strengths from critique."""
        strengths = []
        lines = critique_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['strength', 'good', 'well', 'excellent', 'strong']):
                strengths.append(line.strip())
        
        return strengths[:5]  # Limit to top 5 strengths
    
    def _extract_weaknesses(self, critique_text: str) -> List[str]:
        """Extract weaknesses from critique."""
        weaknesses = []
        lines = critique_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['weakness', 'poor', 'inadequate', 'insufficient', 'weak']):
                weaknesses.append(line.strip())
        
        return weaknesses[:5]  # Limit to top 5 weaknesses
    
    def _extract_priority_improvements(self, recommendations_text: str) -> List[str]:
        """Extract priority improvements from recommendations."""
        improvements = []
        lines = recommendations_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'priority' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                improvements.append(line.strip())
            elif capturing and ('methodological' in line.lower() or 'additional' in line.lower()):
                break
        
        return improvements[:3]  # Top 3 priorities
    
    def _extract_methodological_improvements(self, recommendations_text: str) -> List[str]:
        """Extract methodological improvements from recommendations."""
        improvements = []
        lines = recommendations_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'methodological' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                improvements.append(line.strip())
            elif capturing and ('additional' in line.lower() or 'implementation' in line.lower()):
                break
        
        return improvements
    
    def _extract_additional_experiments(self, recommendations_text: str) -> List[str]:
        """Extract additional experiments from recommendations."""
        experiments = []
        lines = recommendations_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'additional experiments' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                experiments.append(line.strip())
            elif capturing and ('implementation' in line.lower() or 'documentation' in line.lower()):
                break
        
        return experiments
    
    def _extract_implementation_fixes(self, recommendations_text: str) -> List[str]:
        """Extract implementation fixes from recommendations."""
        fixes = []
        lines = recommendations_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'implementation' in line.lower() and 'fix' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                fixes.append(line.strip())
            elif capturing and ('documentation' in line.lower() or 'validation' in line.lower()):
                break
        
        return fixes
    
    def _extract_validation_steps(self, recommendations_text: str) -> List[str]:
        """Extract validation steps from recommendations."""
        steps = []
        lines = recommendations_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'validation' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                steps.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return steps