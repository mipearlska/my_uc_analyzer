"""
Feedback Agent (LLM as Judge)

Uses Groq (llama-3.3-70b) to:
1. Evaluate if the system design satisfies all requirements
2. Provide structured feedback if not
3. Decide whether to approve or request redesign
4. Update learned lessons when design is finalized
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.state import AgentState
from src.agents.memory import LongTermMemory


load_dotenv()


EVALUATION_SYSTEM_PROMPT = """You are a strict technical reviewer evaluating AI system designs.

Your task is to evaluate whether a proposed system design satisfies all the given requirements.

For each requirement, assess:
1. Is it addressed in the design? (Yes/No)
2. How well is it addressed? (Fully/Partially/Not at all)
3. What specific part of the design addresses it?

Be strict but fair. A good design should:
- Explicitly address each requirement
- Have clear agent responsibilities
- Have well-defined data flows
- Integrate properly with network functions

After evaluating all requirements, provide:
1. Overall verdict: APPROVED or NEEDS_REVISION
2. Score: X/Y requirements fully satisfied
3. Detailed feedback for any gaps or improvements needed"""


EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EVALUATION_SYSTEM_PROMPT),
    ("human", """Evaluate the following system design against the requirements.

## Use Case
ID: {use_case_id}
Name: {use_case_name}

## Requirements to Satisfy
{requirements}

## System Design to Evaluate
{system_design}

## Current Iteration
This is iteration {iteration} of {max_iterations}.

Provide your evaluation in this format:

### Requirement Analysis
[For each requirement, state if it's satisfied and how]

### Overall Verdict
[APPROVED or NEEDS_REVISION]

### Score
[X/Y requirements fully satisfied]

### Feedback
[If NEEDS_REVISION, provide specific actionable feedback for improvement]
[If APPROVED, summarize the key strengths of the design]
""")
])


EXTRACT_LESSONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You extract key design lessons from successful AI system designs.

A design lesson is a reusable insight that can help with future designs.
Focus on:
- Architectural patterns that worked well
- Integration approaches with network functions
- Agent interaction patterns
- How specific types of requirements were satisfied

Provide 2-3 concise lessons, each as a single sentence."""),
    ("human", """Extract design lessons from this approved design.

## Use Case
{use_case_name}

## Requirements
{requirements}

## Approved Design
{system_design}

## Feedback Summary
{feedback}

Provide 2-3 key lessons learned (one per line):""")
])


class FeedbackAgent:
    """
    Agent that evaluates designs and provides feedback (LLM as Judge).
    
    Uses Groq (llama-3.3-70b) for evaluation.
    """
    
    def __init__(
        self,
        memory: LongTermMemory,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0  # Deterministic evaluation
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
        
        self.evaluation_chain = EVALUATION_PROMPT | self.llm | StrOutputParser()
        self.lessons_chain = EXTRACT_LESSONS_PROMPT | self.llm | StrOutputParser()
    
    def __call__(self, state: AgentState) -> AgentState:
        """Process the state and return updated state."""
        return self.run(state)
    
    def run(self, state: AgentState) -> AgentState:
        """
        Run the feedback agent.
        
        Steps:
        1. Evaluate design against requirements
        2. Determine if approved or needs revision
        3. If approved or max iterations reached, finalize and extract lessons
        """
        use_case_id = state["use_case_id"]
        use_case_name = state["use_case_name"]
        requirement_list = state["requirement_list"]
        system_design = state["system_design"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        
        # Check for required inputs
        if not system_design or not requirement_list:
            return {
                **state,
                "error": "Feedback agent requires system_design and requirement_list",
                "is_approved": False
            }
        
        try:
            # Step 1: Evaluate design
            evaluation = self.evaluation_chain.invoke({
                "use_case_id": use_case_id,
                "use_case_name": use_case_name,
                "requirements": requirement_list,
                "system_design": system_design,
                "iteration": iteration + 1,
                "max_iterations": max_iterations
            })
            
            # Step 2: Parse verdict
            is_approved = "APPROVED" in evaluation.upper() and "NEEDS_REVISION" not in evaluation.upper()
            
            # Check if we've hit max iterations
            is_final_iteration = (iteration + 1) >= max_iterations
            
            # Step 3: If approved or final iteration, extract lessons and finalize
            if is_approved or is_final_iteration:
                # Extract lessons
                lessons_text = self.lessons_chain.invoke({
                    "use_case_name": use_case_name,
                    "requirements": requirement_list,
                    "system_design": system_design,
                    "feedback": evaluation
                })
                
                # Save lessons to long-term memory
                lessons = [l.strip() for l in lessons_text.strip().split("\n") if l.strip()]
                for lesson in lessons:
                    # Clean up lesson text (remove numbering if present)
                    clean_lesson = lesson.lstrip("0123456789.-) ").strip()
                    if clean_lesson:
                        self.memory.add_lesson(use_case_id, use_case_name, clean_lesson)
                
                # Create final response
                if is_approved:
                    final_status = "✅ Design APPROVED"
                else:
                    final_status = f"⚠️ Design finalized after {max_iterations} iterations (max reached)"
                
                final_response = f"""{final_status}

## Use Case
{use_case_id}: {use_case_name}

## Final System Design
{system_design}

## Evaluation Summary
{evaluation}

## Lessons Learned (saved for future designs)
{lessons_text}
"""
                
                return {
                    **state,
                    "feedback": evaluation,
                    "is_approved": True,  # Mark as complete (either approved or max iterations)
                    "iteration": iteration + 1,
                    "final_response": final_response,
                    "error": None
                }
            
            else:
                # Needs revision - return feedback for Design Agent
                return {
                    **state,
                    "feedback": evaluation,
                    "is_approved": False,
                    "iteration": iteration + 1,
                    "final_response": None,
                    "error": None
                }
            
        except Exception as e:
            return {
                **state,
                "error": f"Feedback evaluation failed: {str(e)}",
                "is_approved": False
            }


# For testing
if __name__ == "__main__":
    from src.agents.memory import LongTermMemory
    from pathlib import Path
    
    # Setup
    print("Setting up...")
    memory = LongTermMemory()
    
    # Create agent
    print("Creating Feedback Agent...")
    agent = FeedbackAgent(memory=memory)
    
    # Test with pre-filled state
    print("\n" + "=" * 60)
    print("TEST: Feedback Agent")
    print("=" * 60)
    
    test_state: AgentState = {
        "user_query": "Design a system for the smart life use case",
        "use_case_id": "5.1.1",
        "use_case_name": "AI Agents to Enable Smart Life",
        "description_summary": "AI agents coordinate smart devices for daily life.",
        "requirement_list": """[PR 5.1.1-1] Subject to operator policy and regulatory requirements, the mobile network 
is used to contact AI-Core, which then provides a mechanism to uniquely identify an AI Agent 
that acts on behalf of the user.

[PR 5.1.1-2] The mobile network uses a combination of authentication, opaque execution, 
and policy enforcement to ensure that all AI Agents preserve the privacy of the owner.

[PR 5.1.1-3] The mobile network provides mechanisms that enable AI Agents to communicate 
with each other while maintaining security and privacy.""",
        "system_design": """## 1. System Overview
A multi-agent system for Smart Life that coordinates embodied AI agents through 6G network services.

## 2. Agents
### Intent Parser Agent
- Role: Parse user intents into structured tasks
- Inputs: Natural language user requests
- Outputs: Structured task definitions
- Tools: NLP parser, intent classifier

### Task Orchestrator Agent
- Role: Decompose tasks and assign to appropriate devices
- Inputs: Structured tasks from Intent Parser
- Outputs: Sub-task assignments
- Tools: Device registry, capability matcher

### Device Coordinator Agent
- Role: Manage communication between embodied AI agents
- Inputs: Network function metrics
- Outputs: Execution commands, status updates
- Tools: Secure messaging, status tracker

## 3. Workflow
1. User sends intent to Intent Parser Agent via mobile network
2. Intent Parser creates structured task, sends to Task Orchestrator
3. Task Orchestrator queries device capabilities, creates sub-tasks

## 4. Network Integration
- AMF: Authentication of AI agents and users
- SMF: Session management for agent communication
- PCF: Policy enforcement for privacy and security
- AI-Core: Central AI agent registry and identification""",
        "feedback": None,
        "is_approved": False,
        "iteration": 0,
        "max_iterations": 3,
        "final_response": None,
        "error": None
    }
    
    print(f"\nUse Case: {test_state['use_case_id']} - {test_state['use_case_name']}")
    print(f"Iteration: {test_state['iteration'] + 1}/{test_state['max_iterations']}")
    print("\nRunning Feedback Agent...")
    
    result = agent.run(test_state)
    
    print(f"\n--- Result ---")
    print(f"Error: {result['error']}")
    print(f"Is Approved: {result['is_approved']}")
    print(f"Iteration: {result['iteration']}")
    print(f"\n{'='*60}")
    print("FEEDBACK:")
    print("="*60)
    print(result['feedback'][:1500] if result['feedback'] else "No feedback")
    
    if result['final_response']:
        print(f"\n{'='*60}")
        print("FINAL RESPONSE:")
        print("="*60)
        print(result['final_response'][:2000])
    
    # Check lessons saved
    print(f"\n{'='*60}")
    print("LESSONS IN MEMORY:")
    print("="*60)
    print(memory.get_all_lessons_summary())