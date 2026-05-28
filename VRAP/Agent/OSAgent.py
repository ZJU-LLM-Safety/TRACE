#!/usr/bin/env python3
"""
OS Agent Abstract Interface
Abstract base class for operating system testing agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from VRAP.container import Container


class OSAgent(ABC):
    """
    Abstract base class for OS Agents
    
    This interface defines the standard methods that all OS agents must implement
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the OS Agent with specified model
        
        Args:
            model_name: Name of the model to use (e.g., 'gpt-4o', 'claude-3', etc.)
        """
        self.model_name = model_name
        
    @abstractmethod
    def run(self, prompt_to_os_agent: str, **kwargs) -> str:
        """
        Run the OS agent with the given prompt
        
        Args:
            prompt_to_os_agent: The prompt/task to execute
            **kwargs: Additional parameters specific to the agent implementation
                - container: Container instance
                - attacker_identity: Identity to execute as
                - environment_init_code: Optional init code
                - item: Task item with metadata
                
        Returns:
            str: History/log of the execution
        """
        pass
    
    @abstractmethod
    def run_verification_process(self, container: Container, item: Dict[str, Any], 
                               history: str, attacker_identity: str, 
                               environment: Optional[Container] = None) -> bool:
        """
        Unified verification processing function
        
        Args:
            container: Container instance to execute commands in
            item: Task item containing evaluation criteria
            history: Execution history to verify against
            attacker_identity: Identity to execute commands as
            environment: Optional environment container for persistent verification
            
        Returns:
            bool: True if verification passed, False otherwise
        """
        pass 
