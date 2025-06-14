from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableConfig


class BaseAgent(ABC):
    """Base class for all research agents."""
    
    def __init__(self, config: Dict[str, Any], agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config["models"]["llm_model"],
            temperature=config["agents"].get(agent_name, {}).get("temperature", 0.3)
        )
        
        self.memory = []
        self.max_iterations = config["agents"].get(agent_name, {}).get("max_iterations", 5)
    
    @abstractmethod
    async def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a request and return results."""
        pass
    
    def add_to_memory(self, message: BaseMessage):
        """Add message to agent memory."""
        self.memory.append(message)
        # Keep memory size manageable
        if len(self.memory) > 20:
            self.memory = self.memory[-15:]  # Keep last 15 messages
    
    def get_memory_context(self) -> str:
        """Get formatted memory context."""
        if not self.memory:
            return ""
        
        context = f"Previous conversation for {self.agent_name}:\n"
        for msg in self.memory[-5:]:  # Last 5 messages
            if isinstance(msg, HumanMessage):
                context += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"Assistant: {msg.content}\n"
        
        return context
    
    async def _llm_call(self, prompt: str, system_prompt: str = "") -> str:
        """Make a call to the LLM with error handling."""
        try:
            messages = []
            
            if system_prompt:
                messages.append(HumanMessage(content=system_prompt))
            
            # Add memory context
            memory_context = self.get_memory_context()
            if memory_context:
                prompt = memory_context + "\n\n" + prompt
            
            messages.append(HumanMessage(content=prompt))
            
            response = await self.llm.ainvoke(messages)
            
            # Add to memory
            self.add_to_memory(HumanMessage(content=prompt))
            self.add_to_memory(AIMessage(content=response.content))
            
            return response.content
        
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return f"Error: Unable to process request due to {str(e)}"
