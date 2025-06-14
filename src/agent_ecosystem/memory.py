from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import logging


class SharedMemory:
    """Shared memory system for inter-agent communication."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory = {
            "findings": [],
            "hypotheses": [],
            "experiments": [],
            "implementations": [],
            "critiques": [],
            "shared_context": {}
        }
        self.agent_states = {}
    
    def store_finding(self, agent_name: str, finding: Dict[str, Any]):
        """Store a research finding."""
        finding_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": "finding",
            "content": finding
        }
        self.memory["findings"].append(finding_entry)
        self.logger.info(f"Stored finding from {agent_name}")
    
    def store_hypothesis(self, agent_name: str, hypothesis: Dict[str, Any]):
        """Store a research hypothesis."""
        hypothesis_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": "hypothesis",
            "content": hypothesis
        }
        self.memory["hypotheses"].append(hypothesis_entry)
        self.logger.info(f"Stored hypothesis from {agent_name}")
    
    def store_experiment(self, agent_name: str, experiment: Dict[str, Any]):
        """Store experiment details and results."""
        experiment_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": "experiment",
            "content": experiment
        }
        self.memory["experiments"].append(experiment_entry)
        self.logger.info(f"Stored experiment from {agent_name}")
    
    def store_implementation(self, agent_name: str, implementation: Dict[str, Any]):
        """Store code implementation."""
        impl_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": "implementation",
            "content": implementation
        }
        self.memory["implementations"].append(impl_entry)
        self.logger.info(f"Stored implementation from {agent_name}")
    
    def store_critique(self, agent_name: str, critique: Dict[str, Any]):
        """Store critique or evaluation."""
        critique_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "type": "critique",
            "content": critique
        }
        self.memory["critiques"].append(critique_entry)
        self.logger.info(f"Stored critique from {agent_name}")
    
    def get_recent_findings(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent findings."""
        return sorted(self.memory["findings"], key=lambda x: x["timestamp"], reverse=True)[:count]
    
    def get_agent_state(self, agent_name: str) -> Dict[str, Any]:
        """Get current state of an agent."""
        return self.agent_states.get(agent_name, {})
    
    def update_agent_state(self, agent_name: str, state: Dict[str, Any]):
        """Update agent state."""
        self.agent_states[agent_name] = state
    
    def get_context_for_agent(self, agent_name: str) -> Dict[str, Any]:
        """Get relevant context for an agent."""
        context = {
            "recent_findings": self.get_recent_findings(3),
            "shared_context": self.memory["shared_context"],
            "agent_state": self.get_agent_state(agent_name)
        }
        
        # Add relevant information based on agent type
        if "research" in agent_name.lower():
            context["recent_hypotheses"] = self.memory["hypotheses"][-3:]
        elif "implementation" in agent_name.lower():
            context["recent_implementations"] = self.memory["implementations"][-3:]
        elif "experiment" in agent_name.lower():
            context["recent_experiments"] = self.memory["experiments"][-3:]
        elif "critique" in agent_name.lower():
            context["recent_critiques"] = self.memory["critiques"][-3:]
        
        return context
    
    def update_shared_context(self, key: str, value: Any):
        """Update shared context that all agents can access."""
        self.memory["shared_context"][key] = value
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of current memory state."""
        return {
            "findings_count": len(self.memory["findings"]),
            "hypotheses_count": len(self.memory["hypotheses"]),
            "experiments_count": len(self.memory["experiments"]),
            "implementations_count": len(self.memory["implementations"]),
            "critiques_count": len(self.memory["critiques"]),
            "active_agents": list(self.agent_states.keys())
        }
