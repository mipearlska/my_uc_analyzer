"""
Design Agent

Uses Groq (llama-3.3-70b) to:
1. Retrieve requirements for the use case
2. Design a pseudo AI agentic system based on:
   - Description summary (from Query Agent)
   - Requirements
   - Learned design lessons (long-term memory)
   - Feedback (if in feedback loop)
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.state import AgentState
from src.agents.memory import LongTermMemory


load_dotenv()


DESIGN_SYSTEM_PROMPT = """You are an AI system architect specializing in designing multi-agent AI systems for telecommunications networks.

Your task is to design a pseudo AI agentic system based on:
1. The use case description summary
2. The requirements that must be satisfied
3. Lessons learned from previous designs (if any)
4. Feedback from previous design iteration (if any)

Your design should include:

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
How the system integrates with 5G/6G network functions (AMF, SMF, UPF, PCF, etc.)

## 5. Requirements Mapping
For each requirement, explain how the design satisfies it.

Be specific and technical. Use proper 3GPP terminology where applicable."""


DESIGN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DESIGN_SYSTEM_PROMPT),
    ("human", """Design an AI agentic system for the following use case.

## Use Case
ID: {use_case_id}
Name: {use_case_name}

## Description Summary
{description_summary}

## Requirements
{requirements}

## Learned Lessons from Previous Designs
{learned_lessons}

## Feedback from Previous Iteration
{feedback}

Please provide your system design:""")
])


class DesignAgent:
    """
    Agent that designs AI agentic systems based on use case requirements.
    
    Uses Groq (llama-3.3-70b) for high-quality design generation.
    """
    
    def __init__(
        self,
        memory: LongTermMemory,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3  # Slight creativity for design
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
        
        self.design_chain = DESIGN_PROMPT | self.llm | StrOutputParser()
    
    def __call__(self, state: AgentState) -> AgentState:
        """Process the state and return updated state."""
        return self.run(state)
    
    def run(self, state: AgentState) -> AgentState:
        """
        Run the design agent.
        
        Steps:
        1. Retrieve requirements for the use case
        2. Get learned lessons from long-term memory
        3. Generate system design
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

            print("------------------DESIGN AGENT REQUIREMENT DEBUG-----------------")
            print(requirements_text)
            print("-------------------------------------------------------------\n")
            
            # Step 2: Get learned lessons from long-term memory
            lessons = self.memory.get_lessons(use_case_id, use_case_name)
            learned_lessons = lessons.get_lessons_text()
            
            # Step 3: Generate design
            design = self.design_chain.invoke({
                "use_case_id": use_case_id,
                "use_case_name": use_case_name,
                "description_summary": description_summary,
                "requirements": requirements_text,
                "learned_lessons": learned_lessons,
                "feedback": feedback
            })
            
            return {
                **state,
                "requirement_list": requirements_text,
                "system_design": design,
                "error": None
            }
            
        except Exception as e:
            return {
                **state,
                "error": f"Design generation failed: {str(e)}",
                "system_design": None
            }


# For testing
if __name__ == "__main__":
    from pathlib import Path
    from src.etsi_parser import parse_etsi_document
    from src.agents.memory import LongTermMemory
    
    # Setup
    print("Setting up...")
    PDF_PATH = Path("data/ETSI_AI-Agent_core_usecases.pdf")
    
    memory = LongTermMemory("./data/lessons.json")
    
    # Create agent
    print("\nCreating Design Agent...")
    agent = DesignAgent(memory=memory)
    
    # Test with pre-filled state (as if Query Agent already ran)
    print("\n" + "=" * 60)
    print("TEST: Design Agent")
    print("=" * 60)
    
    test_description_summary = """
Main Purpose/Goal:
Enhance user experience in mobile core networks by leveraging AI agents that analyze real-time data from multiple sources and dynamically adapt resource allocation strategies.

Key Actors:
1. Users: High-value users with diverse demands and personalized requirements.
2. Devices: User equipment with varying performance capabilities.
3. AI Agents:
   - Monitoring Agents: Stream telemetry data into the Knowledge Graph (KG).
   - Analytics Agent: Executes complex queries to identify network problems impacting high-value customers.
   - AI Orchestrator: Plans and delegates remediation strategies based on analysis results.

Main Service Flows/Interactions:
1. Data Ingestion: Continuous, multi-source data ingestion into the KG from monitoring agents and the BSS/IT Integration Agent.
2. Analysis: Analytics Agent executes complex queries to identify network problems impacting high-value customers.
3. Remediation Planning: AI Orchestrator plans and delegates remediation strategies based on analysis results.
4. Resource Allocation: PCF, SMF, and UPF dynamically adjust resource allocation strategies based on customer attributes, network status, and service experience.
5. Feedback Loop: Monitoring Agents stream telemetry data to verify remediation effectiveness, which is used to refine AI models and improve future interventions.

Overview:
The AI agent-based core network addresses limitations of traditional core networks by enabling real-time analysis and adaptive resource allocation that responds to diverse user demands and evolving network conditions.
"""

    test_requirement_text = """
[PR 5.3.5-1] The system is required to provide authorized Analytics Agents with secure access to a comprehensive, cross-domain dataset. This includes real-time telemetry from network functions as well as relevant customer and service data from the operator's OSS/BSS and IT domains, ideally unified within a query able KG.
[PR 5.3.5-2] The AI Orchestrator is required to be empowered to enact decisions by issuing dynamic configuration and policy adjustments to Core Network Functions (e.g. PCF, SMF). It also needs to be able to manage the lifecycle of the underlying virtualized resources, such as scaling containerized network function instances, to meet service demands.
[PR 5.3.5-3] The system is required to implement a closed-loop assurance mechanism. The AI Orchestrator and Analytics Agent need to continuously verify that optimization actions have successfully improved the user experience and use these performance outcomes as feedback to refine their decision-making models.
"""
    test_state: AgentState = {
        "user_query": "Design a system for the smart life use case",
        "use_case_id": "5.1.1",
        "use_case_name": "AI Agents to Enable Smart Life",
        "description_summary": test_description_summary,
        "requirement_list": test_requirement_text,
        "system_design": None,
        "feedback": None,
        "is_approved": False,
        "iteration": 0,
        "max_iterations": 3,
        "final_response": None,
        "error": None
    }
    
    print(f"\nUse Case: {test_state['use_case_id']} - {test_state['use_case_name']}")
    print("\nRunning Design Agent...")
    
    result = agent.run(test_state)
    
    print(f"\n--- Result ---")
    print(f"Error: {result['error']}")
    print(f"\nRequirements Retrieved: {len(result['requirement_list'] or '')} chars")
    print(f"\n{'='*60}")
    print("SYSTEM DESIGN:")
    print("="*60)
    print(result['system_design'][:2000] if result['system_design'] else "No design generated")
    if result['system_design'] and len(result['system_design']) > 2000:
        print(f"\n... [truncated, total {len(result['system_design'])} chars]")