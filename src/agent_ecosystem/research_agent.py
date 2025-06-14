from .base_agent import BaseAgent
from .memory import SharedMemory
from typing import Dict, Any, Optional, List
import asyncio


class ResearchAgent(BaseAgent):
    """Agent specialized in analyzing research papers and extracting insights."""
    
    def __init__(self, config: Dict[str, Any], shared_memory: SharedMemory):
        super().__init__(config, "research_agent")
        self.shared_memory = shared_memory
        
        # Import RAG components
        from rag_system.retriever import CrossModalRetriever
        self.retriever = CrossModalRetriever(config)
    
    async def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process research request and analyze literature."""
        self.logger.info(f"Processing research request: {request}")
        
        # Get relevant documents from RAG system
        relevant_content = await self.retriever.retrieve_relevant_content(request)
        
        # Analyze the retrieved content
        analysis = await self._analyze_content(request, relevant_content)
        
        # Generate insights and findings
        findings = await self._generate_findings(request, analysis, relevant_content)
        
        # Store findings in shared memory
        self.shared_memory.store_finding("research_agent", findings)
        
        return {
            "status": "completed",
            "agent": "research_agent",
            "request": request,
            "findings": findings,
            "relevant_content": relevant_content
        }
    
    async def _analyze_content(self, request: str, content: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze retrieved content for key insights."""
        
        analysis_prompt = f"""
        Analyze the following research content to answer the question: "{request}"
        
        Content Summary:
        - Documents: {len(content.get('documents', []))} papers found
        - Figures: {len(content.get('figures', []))} figures found
        - Tables: {len(content.get('tables', []))} tables found
        - Equations: {len(content.get('equations', []))} equations found
        
        Content Details:
        {self._format_content_for_analysis(content)}
        
        Please provide:
        1. Key themes and concepts identified
        2. Main research methodologies found
        3. Important findings and results
        4. Gaps or limitations identified
        5. Relationships between different sources
        
        Format your response as a structured analysis.
        """
        
        analysis_text = await self._llm_call(analysis_prompt, 
                                          "You are a research analyst expert at synthesizing academic literature.")
        
        return {
            "analysis_text": analysis_text,
            "content_stats": {
                "total_sources": sum(len(items) for items in content.values()),
                "source_types": list(content.keys())
            }
        }
    
    async def _generate_findings(self, request: str, analysis: Dict[str, Any], content: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate structured findings from the analysis."""
        
        findings_prompt = f"""
        Based on the analysis of research content, generate structured findings for: "{request}"
        
        Analysis: {analysis['analysis_text']}
        
        Please generate findings in the following structure:
        1. Summary: Brief overview of findings
        2. Key Insights: Main discoveries or patterns
        3. Methods Used: Research methodologies identified
        4. Evidence Quality: Assessment of evidence strength
        5. Research Gaps: Areas needing further investigation
        6. Recommendations: Suggested next steps
        
        Provide specific, actionable insights.
        """
        
        findings_text = await self._llm_call(findings_prompt,
                                           "You are a research synthesis expert. Generate clear, structured findings.")
        
        return {
            "summary": self._extract_summary(findings_text),
            "insights": self._extract_insights(findings_text),
            "methods": self._extract_methods(findings_text),
            "evidence_quality": self._assess_evidence_quality(content),
            "gaps": self._identify_gaps(findings_text),
            "recommendations": self._extract_recommendations(findings_text),
            "raw_findings": findings_text
        }
    
    def _format_content_for_analysis(self, content: Dict[str, List[Dict]]) -> str:
        """Format content for LLM analysis."""
        formatted = ""
        
        for content_type, items in content.items():
            if items:
                formatted += f"\n{content_type.upper()}:\n"
                for i, item in enumerate(items[:3]):  # Limit to first 3 items
                    formatted += f"  {i+1}. {item.get('document', item.get('content', 'No content'))[:200]}...\n"
        
        return formatted
    
    def _extract_summary(self, findings_text: str) -> str:
        """Extract summary from findings text."""
        lines = findings_text.split('\n')
        for i, line in enumerate(lines):
            if 'summary' in line.lower():
                # Return next few lines as summary
                summary_lines = lines[i+1:i+4]
                return ' '.join(summary_lines).strip()
        return "Summary not found"
    
    def _extract_insights(self, findings_text: str) -> List[str]:
        """Extract key insights from findings text."""
        insights = []
        lines = findings_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'key insights' in line.lower() or 'insights' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                insights.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return insights[:5]  # Limit to 5 insights
    
    def _extract_methods(self, findings_text: str) -> List[str]:
        """Extract methods from findings text."""
        methods = []
        lines = findings_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'methods' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                methods.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return methods
    
    def _assess_evidence_quality(self, content: Dict[str, List[Dict]]) -> str:
        """Assess quality of evidence from content."""
        total_sources = sum(len(items) for items in content.values())
        
        if total_sources > 10:
            return "High - Multiple sources available"
        elif total_sources > 5:
            return "Medium - Several sources available"
        elif total_sources > 2:
            return "Low - Limited sources available"
        else:
            return "Very Low - Insufficient sources"
    
    def _identify_gaps(self, findings_text: str) -> List[str]:
        """Identify research gaps from findings."""
        gaps = []
        lines = findings_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'gaps' in line.lower() or 'limitations' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                gaps.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return gaps
    
    def _extract_recommendations(self, findings_text: str) -> List[str]:
        """Extract recommendations from findings."""
        recommendations = []
        lines = findings_text.split('\n')
        
        capturing = False
        for line in lines:
            if 'recommendations' in line.lower():
                capturing = True
                continue
            elif capturing and line.startswith(('1.', '2.', '3.', '-', '•')):
                recommendations.append(line.strip())
            elif capturing and line.strip() == '':
                break
        
        return recommendations
