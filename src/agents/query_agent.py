"""
Query Analysis Agent (Agentic RAG)

Uses Ollama (llama3.1) to:
1. Understand user query
2. Identify which use case the user is asking about
3. Call search tool to get description chunks
4. Summarize the description

This is Agentic RAG because the LLM reasons about which
use case to retrieve, rather than blindly embedding the query.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.state import AgentState
from src.vector_store import ChromaVectorStore
from src.data_models import SectionType


# Use case list for the system prompt
USE_CASE_LIST = """
Available Use Cases:

CONSUMER (5.1.x):
- 5.1.1: AI Agents to Enable Smart Life
- 5.1.2: Network-Assisted Collaborative Robots
- 5.1.3: AI Phone

BUSINESS (5.2.x):
- 5.2.1: AI Agent-based Customized Network for Smart City Traffic Monitoring
- 5.2.2: AI Agents-Based Customized Network for Smart Construction Sites
- 5.2.3: AI Agent Ensuring Game Acceleration Experience
- 5.2.4: AI Agent-Assisted Collaborative Energy Distribution in Power Enterprises

OPERATOR (5.3.x):
- 5.3.1: AI Agent-Based Autonomous Network Management
- 5.3.2: AI Agent-Based Disaster Handling Network Management
- 5.3.3: AI Agent-Based Time-Sensitive Network Management
- 5.3.4: AI Agent-Driven Core Network Signalling Optimization
- 5.3.5: AI Agent-Based Core Networks to Enhance User Experience
"""


# Prompt to identify use case from query
IDENTIFY_USE_CASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a use case identification assistant.

Given a user query, identify which use case they are asking about.
     
IMPORTANT DISTINCTION GUIDES:
     If there are two similar potential use cases. Focus on the topic of the use case. Compare User query intent's topic with the use case topic
     For example, 5.2.1 and 5.2.2 are both Agent-based Customized Network but 5.2.1 is for Smart City Traffic Monitorinng, 5.2.2 is for Smart Construction Sites

{use_case_list}

Respond with ONLY the use case ID (e.g., "5.1.1") and nothing else.
If you cannot identify a specific use case, respond with "UNKNOWN".
"""),
    ("human", "{query}")
])


# Prompt to summarize description
SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a technical summarizer.

Summarize the following use case description in maximum 30 sentences.
Focus on:
1. What is the main purpose/goal
2. Who are the key actors (users, devices, AI agents)
3. What are the main service flows or interactions

Be concise and technical."""),
    ("human", """Use Case: {use_case_name}

Description:
{description}

Provide a concise summary:""")
])


class QueryAnalysisAgent:
    """
    Agentic RAG agent that identifies use cases and summarizes descriptions.
    
    Uses Ollama (llama3.1) for reasoning.
    """
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        model: str = "llama3.1:latest",
        temperature: float = 0
    ):
        self.vector_store = vector_store
        self.llm = ChatOllama(model=model, temperature=temperature)
        
        # Create chains
        self.identify_chain = IDENTIFY_USE_CASE_PROMPT | self.llm | StrOutputParser()
        self.summarize_chain = SUMMARIZE_PROMPT | self.llm | StrOutputParser()
        
        # Use case name lookup
        self.use_case_names = {
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
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Process the state and return updated state.
        
        This is the main entry point for LangGraph.
        """
        return self.run(state)
    
    def run(self, state: AgentState) -> AgentState:
        """
        Run the query analysis agent.
        
        Steps:
        1. Identify use case from query
        2. Search for description chunks
        3. Summarize the description
        """
        user_query = state["user_query"]
        
        try:
            #--------------------------------------------------------------------------
            # Option 1: Embeddings_Similarity_comparison RAG
            #--------------------------------------------------------------------------
            use_case_id = identify_use_case_id(user_query=user_query, use_case_dict=self.use_case_names)


            #--------------------------------------------------------------------------
            # Option 2: Simple Agentic RAG, LLM model decides which use case User asks
            #--------------------------------------------------------------------------
            # # Step 1: Identify use case
            # use_case_id = self.identify_chain.invoke({
            #     "use_case_list": USE_CASE_LIST,
            #     "query": user_query
            # }).strip()
            
            # Validate use case ID
            if use_case_id not in self.use_case_names:
                return {
                    **state,
                    "error": f"Could not identify a valid use case from query. Got: {use_case_id}",
                    "use_case_id": None,
                    "use_case_name": None,
                    "description_summary": None
                }
            
            use_case_name = self.use_case_names[use_case_id]
            
            # Step 2: Search for description chunks
            description_chunks = self.vector_store.search(
                query=user_query,
                k=10,
                use_case_id=use_case_id,
                section_type=SectionType.DESCRIPTION
            )
            
            if not description_chunks:
                return {
                    **state,
                    "error": f"No description found for use case {use_case_id}",
                    "use_case_id": use_case_id,
                    "use_case_name": use_case_name,
                    "description_summary": None
                }
            
            # Step 3: Search for requirement chunks
            requirement_chunks = self.vector_store.search(
                query=user_query,
                k=10,
                use_case_id=use_case_id,
                section_type=SectionType.REQUIREMENTS
            )
            
            if not requirement_chunks:
                return {
                    **state,
                    "error": f"No requirement found for use case {use_case_id}",
                    "use_case_id": use_case_id,
                    "use_case_name": use_case_name,
                    "description_summary": None
                }
            
            # Combine description and requirement chunks
            description_text = "\n\n".join([chunk.content for chunk in description_chunks])
            requirement_text = "\n\n".join([chunk.content for chunk in requirement_chunks])

            # print("------------------DESCRIPTION DEBUG--------------------")
            # print(description_text)
            # print("-------------------------------------------------------")
            # print("\n"*2)

            print("------------------REQUIREMENT DEBUG--------------------")
            print(requirement_text)
            print("-------------------------------------------------------")
            print("\n"*2)
            
            # Step 3: Summarize
            summary = self.summarize_chain.invoke({
                "use_case_name": use_case_name,
                "description": description_text
            })
            
            return {
                **state,
                "use_case_id": use_case_id,
                "use_case_name": use_case_name,
                "description_summary": summary,
                "requirement_list": requirement_text,
                "error": None
            }
            
        except Exception as e:
            return {
                **state,
                "error": f"Query analysis failed: {str(e)}",
                "use_case_id": None,
                "use_case_name": None,
                "description_summary": None,
                "requirement_list": None
            }

# =============================================================================
# Helper function to detect which use case user ask about based on embeddings cosine similarity
# =============================================================================

def cosine_similarity(a: list[float], b: list[float]) -> float:
    import numpy as np

    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def identify_use_case_id(
    user_query: str,
    use_case_dict: dict,
) -> tuple[str | None, float]:
    """
    Returns (use_case_id, similarity_score)
    or (None, best_score) if confidence is too low.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    USE_CASE_EMBEDDINGS = {
    uc_id: embeddings.embed_query(name)
    for uc_id, name in use_case_dict.items()
}
    query_embedding = embeddings.embed_query(user_query)

    best_id = None
    best_score = -1.0

    for uc_id, uc_embedding in USE_CASE_EMBEDDINGS.items():
        score = cosine_similarity(query_embedding, uc_embedding)
        if score > best_score:
            best_score = score
            best_id = uc_id

    return best_id

# =============================================================================
# For testing
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from src.etsi_parser import parse_etsi_document
    from src.vector_store import ChromaVectorStore
    
    # Setup
    print("Setting up vector store...")
    PDF_PATH = Path("data/ETSI_AI-Agent_core_usecases.pdf")
    
    store = ChromaVectorStore(persist_directory="./data/chroma_db")
    
    # Check if data exists, if not, load it
    # if store.count() == 0:
    #     print("Loading chunks into vector store...")
    #     chunks = parse_etsi_document(PDF_PATH)
    #     store.add_chunks(chunks)
    
    # print(f"Vector store has {store.count()} chunks")
    
    # Create agent
    print("\nCreating Query Analysis Agent...")
    agent = QueryAnalysisAgent(vector_store=store)
    
    # Test
    print("\n" + "=" * 60)
    print("TEST: Query Analysis Agent")
    print("=" * 60)
    
    test_state: AgentState = {
        "user_query": "Design an AI Agent network for optimizing user service experience",
        "use_case_id": None,
        "use_case_name": None,
        "description_summary": None,
        "requirement_list": None,
        "system_design": None,
        "feedback": None,
        "is_approved": False,
        "iteration": 0,
        "max_iterations": 3,
        "final_response": None,
        "error": None
    }
    
    print(f"\nUser Query: {test_state['user_query']}")
    print("\nRunning agent...")
    
    result = agent.run(test_state)
    
    print(f"\n--- Result ---")
    print(f"Use Case ID: {result['use_case_id']}")
    print(f"Use Case Name: {result['use_case_name']}")
    print(f"Error: {result['error']}")
    print(f"\nDescription Summary:\n{result['description_summary']}")