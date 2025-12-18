"""
ETSI Use Case Design System - CLI Interface

Interactive command-line interface to design AI agentic systems
based on ETSI use cases.

Features:
- Query Analysis Agent (Ollama) - identifies use case
- Design Agent with ReAct + MCP (Groq) - researches and designs
- Feedback Agent (Groq) - evaluates and provides feedback
- Long-term memory for learned lessons
"""

import sys
from pathlib import Path

from src.etsi_parser import parse_etsi_document
from src.vector_store import ChromaVectorStore
from src.agents.memory import LongTermMemory
from src.agents.graph import run_design_workflow


# Use case list for display
USE_CASES = {
    "5.1.1": "AI Agents to Enable Smart Life",
    "5.1.2": "Network-Assisted Collaborative Robots",
    "5.1.3": "AI Phone",
    "5.2.1": "AI Agent-based Customized Network for Smart City Traffic Monitoring",
    "5.2.2": "AI Agents-Based Customized Network for Smart Construction Sites",
    "5.2.3": "AI Agent Ensuring Game Acceleration Experience",
    "5.2.4": "AI Agent-Assisted Collaborative Energy Distribution in Power Enterprises",
    "5.3.1": "AI Agent-Based Autonomous Network Management",
    "5.3.2": "AI Agent-Based Disaster Handling Network Management",
    "5.3.3": "AI Agent-Based Time-Sensitive Network Management",
    "5.3.4": "AI Agent-Driven Core Network Signalling Optimization",
    "5.3.5": "AI Agent-Based Core Networks to Enhance User Experience",
}


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║             ETSI Use Case Design System                          ║
║             ────────────────────────────                         ║
║   Multi-Agent System with:                                       ║
║   • Query Agent (Ollama) - Use case identification               ║
║   • Design Agent (Groq + MCP Web Search) - System design         ║
║   • Feedback Agent (Groq) - Design evaluation                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def print_help():
    """Print help message."""
    print("""
Commands:
─────────────────────────────────────────────────────────────────────
  <query>     Design a system (e.g., "Design a system for smart life")
  
  list        Show available use cases
  lessons     Show learned design lessons
  clear       Clear all learned lessons
  help        Show this help message
  quit/exit   Exit the program
─────────────────────────────────────────────────────────────────────

Example queries:
  • "Design an AI system for the smart life use case"
  • "Create a multi-agent system for disaster handling"
  • "Build an agentic system for game acceleration"
    """)


def print_use_cases():
    """Print available use cases."""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║                    AVAILABLE USE CASES                           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    print("\n📱 CONSUMER USE CASES (5.1.x)")
    print("─" * 50)
    for uc_id, name in USE_CASES.items():
        if uc_id.startswith("5.1"):
            print(f"   {uc_id}: {name}")
    
    print("\n🏢 BUSINESS USE CASES (5.2.x)")
    print("─" * 50)
    for uc_id, name in USE_CASES.items():
        if uc_id.startswith("5.2"):
            print(f"   {uc_id}: {name}")
    
    print("\n📡 OPERATOR USE CASES (5.3.x)")
    print("─" * 50)
    for uc_id, name in USE_CASES.items():
        if uc_id.startswith("5.3"):
            print(f"   {uc_id}: {name}")
    
    print()


def print_lessons(memory: LongTermMemory):
    """Print learned lessons."""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║                  LEARNED DESIGN LESSONS                          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(memory.get_all_lessons_summary())
    print()


def initialize_system():
    """Initialize vector store and memory."""
    print("🔄 Initializing system...")
    
    PDF_PATH = Path("data/ETSI_AI-Agent_core_usecases.pdf")
    
    # Initialize vector store
    store = ChromaVectorStore()
    
    # Load chunks if needed
    if store.count() == 0:
        if not PDF_PATH.exists():
            print(f"\n❌ Error: PDF not found at {PDF_PATH}")
            print("   Please place the ETSI PDF in the data/ folder.")
            return None, None
        
        print("   Loading ETSI document into vector store...")
        chunks = parse_etsi_document(PDF_PATH)
        store.add_chunks(chunks)
    
    print(f"   ✓ Vector store ready ({store.count()} chunks)")
    
    # Initialize memory
    memory = LongTermMemory()
    print("   ✓ Long-term memory ready")
    
    return store, memory


def run_query(user_input: str, store: ChromaVectorStore, memory: LongTermMemory):
    """Run the design workflow for a user query."""
    print("\n" + "═" * 70)
    print("🚀 STARTING DESIGN WORKFLOW")
    print("═" * 70)
    
    try:
        result = run_design_workflow(
            user_query=user_input,
            vector_store=store,
            memory=memory,
            max_iterations=3,
            verbose=True
        )
        
        # Print result
        print("\n" + "═" * 70)
        print("📋 FINAL RESULT")
        print("═" * 70)
        
        if result.get("error"):
            print(f"\n❌ Error: {result['error']}")
        elif result.get("final_response"):
            print(result["final_response"])
        else:
            print("\n⚠️ No response generated")
            if result.get("system_design"):
                print("\nPartial design was created:")
                print("-" * 50)
                print(result["system_design"][:1500])
                if len(result["system_design"]) > 1500:
                    print(f"\n... [truncated, total {len(result['system_design'])} chars]")
                    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI loop."""
    print_banner()
    
    # Initialize
    store, memory = initialize_system()
    if not store:
        sys.exit(1)
    
    print_help()
    
    while True:
        try:
            print("\n" + "─" * 70)
            user_input = input("🎯 Enter query (or 'help'): ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            cmd = user_input.lower()
            
            if cmd in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            elif cmd == "help":
                print_help()
                continue
            
            elif cmd == "list":
                print_use_cases()
                continue
            
            elif cmd == "lessons":
                print_lessons(memory)
                continue
            
            elif cmd == "clear":
                confirm = input("⚠️ Clear all lessons? (yes/no): ").strip().lower()
                if confirm == "yes":
                    lessons_path = Path("./data/lessons.json")
                    if lessons_path.exists():
                        lessons_path.unlink()
                    memory = LongTermMemory("./data/lessons.json")
                    print("✓ Lessons cleared")
                else:
                    print("Cancelled")
                continue
            
            # Run design workflow
            run_query(user_input, store, memory)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()