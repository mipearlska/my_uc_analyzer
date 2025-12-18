"""
LangGraph Workflow for ETSI Use Case Design System.

Connects:
1. Query Analysis Agent (Ollama) - identifies use case, summarizes description
2. Design Agent (Groq) - designs AI agentic system
3. Feedback Agent (Groq) - evaluates design, provides feedback

With feedback loop (max 3 iterations) between Design and Feedback agents.
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.query_agent import QueryAnalysisAgent
from src.agents.design_agent_react import DesignAgentReAct
from src.agents.feedback_agent import FeedbackAgent
from src.agents.memory import LongTermMemory
from src.vector_store import ChromaVectorStore

from dotenv import load_dotenv

load_dotenv()


def create_workflow(
    vector_store: ChromaVectorStore,
    memory: LongTermMemory,
    max_iterations: int = 3
) -> StateGraph:
    """
    Create the LangGraph workflow.
    
    Graph structure:
    
        ┌─────────────────┐
        │  START          │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Query Agent    │
        │  (Ollama)       │
        └────────┬────────┘
                 │
                 ▼
            ┌────────┐
            │ Error? │──── Yes ──▶ END (with error)
            └────┬───┘
                 │ No
                 ▼
        ┌─────────────────┐
        │  Design Agent   │◀─────────────┐
        │  (Groq)         │              │
        └────────┬────────┘              │
                 │                       │
                 ▼                       │
        ┌─────────────────┐              │
        │ Feedback Agent  │              │
        │  (Groq/Judge)   │              │
        └────────┬────────┘              │
                 │                       │
                 ▼                       │
            ┌──────────┐                 │
            │Approved? │── No ───────────┘
            │or Max?   │   (feedback loop)
            └────┬─────┘
                 │ Yes
                 ▼
        ┌─────────────────┐
        │      END        │
        └─────────────────┘
    """
    
    # Create agents
    query_agent = QueryAnalysisAgent(vector_store=vector_store)
    design_agent = DesignAgentReAct(memory=memory)
    feedback_agent = FeedbackAgent(memory=memory)
    
    # Define node functions
    def query_node(state: AgentState) -> AgentState:
        """Run query analysis agent."""
        result = query_agent(state)
        result["max_iterations"] = max_iterations
        return result
    
    def design_node(state: AgentState) -> AgentState:
        """Run design agent."""
        return design_agent(state)
    
    def feedback_node(state: AgentState) -> AgentState:
        """Run feedback agent."""
        return feedback_agent(state)
    
    # Define routing functions
    def route_after_query(state: AgentState) -> Literal["design", "end"]:
        """Route after query agent - check for errors."""
        if state.get("error"):
            return "end"
        if not state.get("use_case_id"):
            return "end"
        return "design"
    
    def route_after_feedback(state: AgentState) -> Literal["design", "end"]:
        """Route after feedback - continue loop or end."""
        if state.get("error"):
            return "end"
        if state.get("is_approved"):
            return "end"
        # is_approved is False, but we need to check if we should continue
        # The feedback agent sets is_approved=True when max iterations reached
        return "end" if state.get("final_response") else "design"
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query", query_node)
    workflow.add_node("design", design_node)
    workflow.add_node("feedback", feedback_node)
    
    # Add edges
    workflow.set_entry_point("query")
    
    workflow.add_conditional_edges(
        "query",
        route_after_query,
        {
            "design": "design",
            "end": END
        }
    )
    
    workflow.add_edge("design", "feedback")
    
    workflow.add_conditional_edges(
        "feedback",
        route_after_feedback,
        {
            "design": "design",
            "end": END
        }
    )
    
    return workflow


def create_app(
    vector_store: ChromaVectorStore,
    memory: LongTermMemory,
    max_iterations: int = 3
):
    """Create compiled workflow app."""
    workflow = create_workflow(vector_store, memory, max_iterations)
    return workflow.compile()


def run_design_workflow(
    user_query: str,
    vector_store: ChromaVectorStore,
    memory: LongTermMemory,
    max_iterations: int = 3,
    verbose: bool = True
) -> AgentState:
    """
    Run the complete design workflow.
    
    Args:
        user_query: User's request (e.g., "Design a system for smart life use case")
        vector_store: ChromaVectorStore with ETSI chunks
        memory: LongTermMemory for design lessons
        max_iterations: Maximum feedback loop iterations
        verbose: Print progress
        
    Returns:
        Final AgentState with design and feedback
    """
    
    # Create app
    app = create_app(vector_store, memory, max_iterations)
    
    # Initial state
    initial_state: AgentState = {
        "user_query": user_query,
        "use_case_id": None,
        "use_case_name": None,
        "description_summary": None,
        "requirement_list": None,
        "system_design": None,
        "research_search": None,
        "feedback": None,
        "is_approved": False,
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_response": None,
        "error": None
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("ETSI Use Case Design Workflow")
        print(f"{'='*60}")
        print(f"\nUser Query: {user_query}")
        print(f"Max Iterations: {max_iterations}")
        print("\nRunning workflow...\n")
    
    # Run workflow
    final_state = None
    for step, state in enumerate(app.stream(initial_state)):
        # state is a dict with node name as key
        node_name = list(state.keys())[0]
        node_state = state[node_name]
        
        if verbose:
            print(f"Step {step + 1}: {node_name}")
            
            if node_name == "query":
                if node_state.get("error"):
                    print(f"  ❌ Error: {node_state['error']}")
                else:
                    print(f"  ✓ Identified: {node_state.get('use_case_id')} - {node_state.get('use_case_name')}")
            
            elif node_name == "design":
                if node_state.get("error"):
                    print(f"  ❌ Error: {node_state['error']}")
                else:
                    design_len = len(node_state.get('system_design') or '')
                    print(f"  ✓ Design generated ({design_len} chars)")
            
            elif node_name == "feedback":
                if node_state.get("error"):
                    print(f"  ❌ Error: {node_state['error']}")
                else:
                    iteration = node_state.get('iteration', 0)
                    is_approved = node_state.get('is_approved', False)
                    status = "APPROVED" if is_approved else "NEEDS_REVISION"
                    print(f"  ✓ Iteration {iteration}: {status}")
            
            print()
        
        final_state = node_state
    
    return final_state


# For testing
if __name__ == "__main__":
    
    # Setup
    print("Setting up...")
    
    store = ChromaVectorStore()
    memory = LongTermMemory()
    
    # Check if data exists
    if store.count() == 0:
        print("No data in VectorDB...")
    
    print(f"Vector store has {store.count()} chunks")
    
    # Run workflow
    result = run_design_workflow(
        user_query="Design a system for the AI Agents to Enable Smart Life use case",
        vector_store=store,
        memory=memory,
        max_iterations=3,
        verbose=True
    )
    
    # Print final result
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    
    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
    elif result.get("final_response"):
        print(result["final_response"])
    else:
        print("\n⚠️ Workflow ended without final response")
        print(f"State: is_approved={result.get('is_approved')}, iteration={result.get('iteration')}")