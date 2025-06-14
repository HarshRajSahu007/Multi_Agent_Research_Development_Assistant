from typing import Dict, Any, List, Optional
import asyncio
import logging
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver

from .base_agent import BaseAgent
from .memory import SharedMemory
from .research_agent import ResearchAgent
from .implementation_agent import ImplementationAgent
from .experiment_agent import ExperimentAgent
from .critique_agent import CritiqueAgent


class OrchestratorAgent:
    """Orchestrates the multi-agent system using LangGraph."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize shared memory
        self.shared_memory = SharedMemory()
        
        # Initialize specialized agents
        self.research_agent = ResearchAgent(config, self.shared_memory)
        self.implementation_agent = ImplementationAgent(config, self.shared_memory)
        self.experiment_agent = ExperimentAgent(config, self.shared_memory)
        self.critique_agent = CritiqueAgent(config, self.shared_memory)
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for agent coordination."""
        
        # Define the workflow state
        class WorkflowState:
            def __init__(self):
                self.query: str = ""
                self.context: Dict[str, Any] = {}
                self.results: Dict[str, Any] = {}
                self.current_step: str = ""
                self.completed_steps: List[str] = []
                self.final_response: Dict[str, Any] = {}
        
        # Create the workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("research", self._research_node)
        workflow.add_node("implementation", self._implementation_node)
        workflow.add_node("experiment", self._experiment_node)
        workflow.add_node("critique", self._critique_node)
        workflow.add_node("synthesis", self._synthesis_node)
        
        # Define the workflow flow
        workflow.set_entry_point("research")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "research",
            self._should_implement,
            {
                "implement": "implementation",
                "experiment": "experiment",
                "critique": "critique"
            }
        )
        
        workflow.add_conditional_edges(
            "implementation",
            self._should_experiment,
            {
                "experiment": "experiment",
                "critique": "critique"
            }
        )
        
        workflow.add_conditional_edges(
            "experiment",
            self._should_critique,
            {
                "critique": "critique",
                "synthesis": "synthesis"
            }
        )
        
        workflow.add_edge("critique", "synthesis")
        workflow.add_edge("synthesis", END)
        
        return workflow
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query through the multi-agent workflow."""
        self.logger.info(f"Orchestrating query: {query}")
        
        # Initialize workflow state
        state = self.workflow.get_state_schema()()
        state.query = query
        state.context = context or {}
        
        # Execute the workflow
        try:
            # For this implementation, we'll execute sequentially
            # In a full LangGraph implementation, this would be handled by the graph
            final_result = await self._execute_sequential_workflow(query, context)
            return final_result
        
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    async def _execute_sequential_workflow(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute workflow sequentially (simplified version)."""
        results = {
            "query": query,
            "steps_completed": [],
            "results": {}
        }
        
        # Step 1: Research
        self.logger.info("Step 1: Research Analysis")
        research_result = await self.research_agent.process_request(query, context)
        results["results"]["research"] = research_result
        results["steps_completed"].append("research")
        
        # Step 2: Implementation (if applicable)
        if self._query_needs_implementation(query):
            self.logger.info("Step 2: Implementation")
            impl_result = await self.implementation_agent.process_request(query, context)
            results["results"]["implementation"] = impl_result
            results["steps_completed"].append("implementation")
        
        # Step 3: Experiment (if applicable)
        if self._query_needs_experiment(query):
            self.logger.info("Step 3: Experiment Design & Execution")
            exp_result = await self.experiment_agent.process_request(query, context)
            results["results"]["experiment"] = exp_result
            results["steps_completed"].append("experiment")
        
        # Step 4: Critique
        self.logger.info("Step 4: Critical Analysis")
        critique_result = await self.critique_agent.process_request(query, context)
        results["results"]["critique"] = critique_result
        results["steps_completed"].append("critique")
        
        # Step 5: Synthesis
        self.logger.info("Step 5: Synthesis")
        synthesis_result = await self._synthesize_results(query, results["results"])
        results["synthesis"] = synthesis_result
        results["steps_completed"].append("synthesis")
        
        results["status"] = "completed"
        results["final_response"] = synthesis_result
        
        return results
    
    async def _research_node(self, state) -> Dict[str, Any]:
        """Research node for LangGraph workflow."""
        result = await self.research_agent.process_request(state.query, state.context)
        state.results["research"] = result
        state.completed_steps.append("research")
        state.current_step = "research_completed"
        return state
    
    async def _implementation_node(self, state) -> Dict[str, Any]:
        """Implementation node for LangGraph workflow."""
        result = await self.implementation_agent.process_request(state.query, state.context)
        state.results["implementation"] = result
        state.completed_steps.append("implementation")
        state.current_step = "implementation_completed"
        return state
    
    async def _experiment_node(self, state) -> Dict[str, Any]:
        """Experiment node for LangGraph workflow."""
        result = await self.experiment_agent.process_request(state.query, state.context)
        state.results["experiment"] = result
        state.completed_steps.append("experiment")
        state.current_step = "experiment_completed"
        return state
    
    async def _critique_node(self, state) -> Dict[str, Any]:
        """Critique node for LangGraph workflow."""
        result = await self.critique_agent.process_request(state.query, state.context)
        state.results["critique"] = result
        state.completed_steps.append("critique")
        state.current_step = "critique_completed"
        return state
    
    async def _synthesis_node(self, state) -> Dict[str, Any]:
        """Synthesis node for LangGraph workflow."""
        synthesis = await self._synthesize_results(state.query, state.results)
        state.final_response = synthesis
        state.current_step = "completed"
        return state
    
    def _should_implement(self, state) -> str:
        """Determine if implementation is needed."""
        if self._query_needs_implementation(state.query):
            return "implement"
        elif self._query_needs_experiment(state.query):
            return "experiment"
        else:
            return "critique"
    
    def _should_experiment(self, state) -> str:
        """Determine if experimentation is needed."""
        if self._query_needs_experiment(state.query):
            return "experiment"
        else:
            return "critique"
    
    def _should_critique(self, state) -> str:
        """Determine if critique is needed (always yes)."""
        return "critique"
    
    def _query_needs_implementation(self, query: str) -> bool:
        """Determine if query needs implementation."""
        implementation_keywords = [
            "implement", "code", "algorithm", "build", "create", "develop",
            "program", "software", "application", "system", "tool"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in implementation_keywords)
    
    def _query_needs_experiment(self, query: str) -> bool:
        """Determine if query needs experimentation."""
        experiment_keywords = [
            "test", "experiment", "validate", "evaluate", "compare", "measure",
            "benchmark", "performance", "accuracy", "effectiveness", "trial"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in experiment_keywords)
    
    async def _synthesize_results(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from all agents into a comprehensive response."""
        
        # Extract key information from each agent's results
        research_summary = ""
        if "research" in results:
            research_data = results["research"].get("findings", {})
            research_summary = research_data.get("summary", "No research summary available")
        
        implementation_summary = ""
        if "implementation" in results:
            impl_data = results["implementation"].get("implementation", {})
            implementation_summary = f"Implementation completed: {impl_data.get('request', 'No description')}"
        
        experiment_summary = ""
        if "experiment" in results:
            exp_data = results["experiment"].get("experiment", {})
            experiment_summary = f"Experiment conducted: {exp_data.get('request', 'No description')}"
        
        critique_summary = ""
        if "critique" in results:
            critique_data = results["critique"].get("critique", {})
            critique_summary = f"Critique completed with recommendations provided"
        
        # Create comprehensive synthesis
        synthesis = {
            "query": query,
            "executive_summary": f"Comprehensive analysis completed for: {query}",
            "key_findings": [],
            "recommendations": [],
            "next_steps": [],
            "confidence_score": 0.0,
            "completeness_score": 0.0
        }
        
        # Aggregate findings
        if "research" in results:
            research_findings = results["research"].get("findings", {})
            if research_findings.get("insights"):
                synthesis["key_findings"].extend(research_findings["insights"][:3])
        
        # Aggregate recommendations
        if "critique" in results:
            critique_recs = results["critique"].get("critique", {}).get("recommendations", {})
            if critique_recs.get("priority_improvements"):
                synthesis["recommendations"].extend(critique_recs["priority_improvements"][:3])
        
        # Calculate confidence and completeness scores
        steps_completed = len(results)
        max_steps = 4  # research, implementation, experiment, critique
        synthesis["completeness_score"] = (steps_completed / max_steps) * 100
        
        # Simple confidence calculation based on agent results
        confidence_factors = []
        if "research" in results:
            confidence_factors.append(0.8)  # Research baseline
        if "implementation" in results:
            impl_validation = results["implementation"].get("implementation", {}).get("validation", {})
            if impl_validation.get("syntax_valid"):
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.5)
        if "experiment" in results:
            confidence_factors.append(0.7)  # Experiment baseline
        if "critique" in results:
            confidence_factors.append(0.6)  # Critique baseline
        
        if confidence_factors:
            synthesis["confidence_score"] = sum(confidence_factors) / len(confidence_factors) * 100
        
        # Generate next steps
        synthesis["next_steps"] = [
            "Review and validate all findings",
            "Consider additional experiments if needed",
            "Implement recommended improvements",
            "Document and share results"
        ]
        
        return synthesis
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of shared memory state."""
        return self.shared_memory.get_memory_summary()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            "research_agent": "active",
            "implementation_agent": "active", 
            "experiment_agent": "active",
            "critique_agent": "active",
            "orchestrator": "active",
            "shared_memory": self.get_memory_summary()
        }