"""
State definition for the LangGraph agent workflow.

This defines the short-term memory that flows between agents.
"""

from typing import TypedDict, Optional, Annotated
from operator import add


class AgentState(TypedDict):
    """
    State passed between agents in the workflow.
    
    Short-term memory (within one workflow run):
    - user_query: Original user question
    - use_case_id: Identified use case (e.g., "5.1.1")
    - use_case_name: Use case title
    - description_summary: Summary from Query Agent
    - requirement_list: Requirements retrieved by Design Agent
    - system_design: Current design from Design Agent
    - feedback: Feedback from Feedback Agent
    - is_approved: Whether design is approved
    - iteration: Current feedback loop iteration
    - final_response: Final response to user
    - error: Any error message
    """
    
    # Input
    user_query: str
    
    # From Query Analysis Agent
    use_case_id: Optional[str]
    use_case_name: Optional[str]
    description_summary: Optional[str]
    
    # From Design Agent
    requirement_list: Optional[str]
    system_design: Optional[str]
    research_search: Optional[str]
    
    # From Feedback Agent
    feedback: Optional[str]
    is_approved: bool
    
    # Control flow
    iteration: int
    max_iterations: int
    
    # Output
    final_response: Optional[str]
    error: Optional[str]