"""
Design Agent with ReAct Pattern and MCP Tools.

Uses Groq (llama-3.3-70b) with:
- ReAct pattern: Thought → Action → Observation loop
- MCP Tools: Web research for latest specifications
- Retrieval: Requirements from vector store

The agent autonomously decides when to search for additional information.
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent


from src.agents.state import AgentState
from src.agents.memory import LongTermMemory
from src.agents.mcp_tools import research
from src.vector_store import ChromaVectorStore
from src.data_models import SectionType


load_dotenv()


DESIGN_SYSTEM_PROMPT = """You are an AI system architect specializing in designing multi-agent AI systems for telecommunications networks.

Your task is to design a pseudo AI agentic system based on the use case information provided.

## Available Tools

You have access to a research tool that searches the web and summarizes findings. Use it to:
- Find latest 3GPP/ETSI specifications relevant to the design
- Research AI agent architectural patterns
- Look up telecom network function details (NWDAF, NEF, AMF, SMF, etc.)
- Find multi-agent system best practices

## When to Use Research Tool

Use the research tool when you need:
- Current standards or specifications (your training data may be outdated)
- Specific technical details about 5G/6G network functions
- Best practices for a particular pattern you want to use

Limit yourself to 1-2 research queries maximum to keep focused.

## Design Output Format

After your research (if any), provide a complete design with:

## 1. System Overview
Brief description of the overall system architecture.

## 2. Agents
List each AI agent with:
- Agent name
- Role/responsibility  
- Inputs it receives
- Outputs it produces
- Tools it can use

## 3. Workflow
Describe how agents interact, including:
- Sequence of operations
- Data flow between agents
- Decision points

## 4. Network Integration
How the system integrates with 5G/6G network functions (AMF, SMF, UPF, PCF, NWDAF, NEF, etc.)

## 5. Requirements Mapping
For each requirement, explain how the design satisfies it.

Be specific and technical. Use proper 3GPP terminology."""


class DesignAgentReAct:
    """
    Design Agent using ReAct pattern with MCP tools.
    
    The agent can autonomously decide to:
    1. Research latest specifications using web search
    2. Look up architectural patterns
    3. Then generate the design with enriched context
    """
    
    def __init__(
        self,
        memory: LongTermMemory,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3
    ):
        self.memory = memory
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        
        # Tools available to the agent
        self.tools = [research]
        
        # Create the ReAct agent using langgraph
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools
        )
    
    def __call__(self, state: AgentState) -> AgentState:
        """Process the state and return updated state."""
        return self.run(state)
    
    def run(self, state: AgentState) -> AgentState:
        """
        Run the ReAct design agent.
        
        Steps:
        1. Retrieve requirements for the use case
        2. Get learned lessons from long-term memory
        3. Run agent (may do research via tools)
        4. Return final design
        """
        use_case_id = state["use_case_id"]
        use_case_name = state["use_case_name"]
        description_summary = state["description_summary"]
        feedback = state.get("feedback") or "No feedback yet - this is the first design iteration."
        
        # Check for required inputs
        if not use_case_id or not description_summary:
            return {
                **state,
                "error": "Design agent requires use_case_id and description_summary",
                "system_design": None
            }
        
        try:
            # Step 1: Retrieve requirements
            requirements_text = state["requirement_list"]

            
            # Step 2: Get learned lessons from long-term memory
            lessons = self.memory.get_lessons(use_case_id, use_case_name)
            learned_lessons = lessons.get_lessons_text()
            
            # Step 3: Build the user message with all context
            if not state["research_search"]:
                user_message = f"""Design an AI agentic system for the following use case.

    ## Use Case
    ID: {use_case_id}
    Name: {use_case_name}

    ## Description Summary
    {description_summary}

    ## Requirements
    {requirements_text}

    ## Learned Lessons from Previous Designs
    {learned_lessons}

    ## Feedback from Previous Iteration
    {feedback}

    First, Use the research tool to search for best practices related to user request content.
    Then, provide your complete system design following the format specified."""
            else:
                user_message = f"""Design an AI agentic system for the following use case.

    ## Use Case
    ID: {use_case_id}
    Name: {use_case_name}

    ## Description Summary
    {description_summary}

    ##Research Summary
    {state["research_search"]}

    ## Requirements
    {requirements_text}

    ## Learned Lessons from Previous Designs
    {learned_lessons}

    ## Feedback from Previous Iteration
    {feedback}

    Provide your complete system design following the format specified."""
            

            # Step 4: Run the ReAct agent
            messages = [
                SystemMessage(content=DESIGN_SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ]
            
            # Invoke the agent
            result = self.agent.invoke({"messages": messages})
            
            # Extract the final response
            final_messages = result.get("messages", [])
            design = ""
            research_queries = []
            research_summaries = ""

            tool_call_ids = {}
            
            for msg in final_messages:
                # Collect tool calls (research queries)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get('name') == 'research':
                            query = tool_call.get('args', {}).get('query', '')
                            tool_id = tool_call.get('id', '')
                            if query:
                                research_queries.append(query)
                                tool_call_ids[tool_id] = query

                # Collect tool responses (research summaries) - NEW
                if msg.type == 'tool' and hasattr(msg, 'tool_call_id'):
                    tool_id = msg.tool_call_id
                    if tool_id in tool_call_ids:
                        research_summaries += msg.content
                
                # Get the final AI response
                if hasattr(msg, 'content') and msg.content:
                    if msg.type == 'ai' and not hasattr(msg, 'tool_calls'):
                        design = msg.content
                    elif msg.type == 'ai' and hasattr(msg, 'tool_calls') and not msg.tool_calls:
                        design = msg.content
            
            # If design is still empty, get the last AI message
            if not design:
                for msg in reversed(final_messages):
                    if hasattr(msg, 'content') and msg.content and msg.type == 'ai':
                        design = msg.content
                        break
            
            # Add research summary if research was done
            if research_queries:
                research_call = "## Research Conducted\nThe agent researched the following topics:\n"
                for q in research_queries:
                    research_call += f"- \"{q}\"\n"
                research_call += "\n---\n\n"
                design = research_call + design
            
            return {
                **state,
                "requirement_list": requirements_text,
                "research_search": research_summaries,
                "system_design": design,
                "error": None
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                **state,
                "error": f"Design generation failed: {str(e)}",
                "system_design": None
            }


# For testing
if __name__ == "__main__":
    
    # Setup
    print("Setting up...")

    memory = LongTermMemory("./data/lessons.json")
    
    # Create agent
    print("\nCreating ReAct Design Agent with MCP tools...")
    agent = DesignAgentReAct(memory)
    
    # Test with pre-filled state (as if Query Agent already ran)
    print("\n" + "=" * 60)
    print("TEST: ReAct Design Agent with MCP Web Search")
    print("=" * 60)
    
    test_state: AgentState = {
        "user_query": "Design a system for the smart life use case",
        "use_case_id": "5.1.1",
        "use_case_name": "AI Agents to Enable Smart Life",
        "description_summary": """The Smart Life use case enables AI agents to coordinate 
smart devices (robot-servants, smart cars, drones) to assist users in daily life. 
The user expresses intents (like "make a camping plan") which are parsed by network 
AI agents into sub-tasks distributed to appropriate devices. Key actors include 
the user, embodied AI agents (robots, cars), and network AI services. The service 
flow involves intent parsing, task decomposition, agent coordination, and result synthesis.""",
        "requirement_list": """
[PR 5.3.5-1] The system is required to provide authorized Analytics Agents with secure access to a comprehensive, cross-domain dataset. This includes real-time telemetry from network functions as well as relevant customer and service data from the operator's OSS/BSS and IT domains, ideally unified within a query able KG.
[PR 5.3.5-2] The AI Orchestrator is required to be empowered to enact decisions by issuing dynamic configuration and policy adjustments to Core Network Functions (e.g. PCF, SMF). It also needs to be able to manage the lifecycle of the underlying virtualized resources, such as scaling containerized network function instances, to meet service demands.
[PR 5.3.5-3] The system is required to implement a closed-loop assurance mechanism. The AI Orchestrator and Analytics Agent need to continuously verify that optimization actions have successfully improved the user experience and use these performance outcomes as feedback to refine their decision-making models.
""",
        "system_design": None,
        "feedback": None,
        "is_approved": False,
        "iteration": 0,
        "max_iterations": 3,
        "final_response": None,
        "error": None
    }
    
    print(f"\nUse Case: {test_state['use_case_id']} - {test_state['use_case_name']}")
    print("\nRunning ReAct Design Agent...")
    print("-" * 60)
    
    result = agent.run(test_state)
    
    print("-" * 60)
    print(f"\n--- Result ---")
    print(f"Error: {result['error']}")
    print(f"\nRequirements Retrieved: {len(result['requirement_list'] or '')} chars")
    print(f"\n{'='*60}")
    print("SYSTEM DESIGN:")
    print("="*60)
    if result['system_design']:
        print(result['system_design'][:3000])
        if len(result['system_design']) > 3000:
            print(f"\n... [truncated, total {len(result['system_design'])} chars]")
    else:
        print("No design generated")