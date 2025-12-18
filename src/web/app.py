"""
Streamlit frontend for ETSI Use Case Design System.
"""

import time
import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="ETSI Use Case Design System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# API Functions
# =============================================================================

def api_health():
    """Check API health."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.ok else None
    except:
        return None


def api_get_use_cases():
    """Get list of use cases."""
    try:
        response = requests.get(f"{API_BASE_URL}/use-cases", timeout=5)
        return response.json() if response.ok else None
    except:
        return None


def api_start_design(query: str, max_iterations: int = 3):
    """Start a design workflow."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/design",
            json={"query": query, "max_iterations": max_iterations},
            timeout=10
        )
        return response.json() if response.ok else None
    except Exception as e:
        return {"error": str(e)}


def api_get_design_status(task_id: str):
    """Get design task status."""
    try:
        response = requests.get(f"{API_BASE_URL}/design/{task_id}", timeout=10)
        return response.json() if response.ok else None
    except:
        return None


def api_get_lessons():
    """Get all lessons."""
    try:
        response = requests.get(f"{API_BASE_URL}/lessons", timeout=5)
        return response.json() if response.ok else None
    except:
        return None


def api_clear_lessons():
    """Clear all lessons."""
    try:
        response = requests.delete(f"{API_BASE_URL}/lessons", timeout=5)
        return response.ok
    except:
        return False


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render sidebar with use cases and settings."""
    with st.sidebar:
        st.title("🤖 ETSI Design System")
        st.markdown("---")
        
        # Health check
        health = api_health()
        if health:
            st.success(f"✓ API Connected")
            st.caption(f"Vector store: {health['vector_store_count']} chunks")
            st.caption(f"Lessons: {health['lessons_count']}")
        else:
            st.error("✗ API Offline")
            st.caption("Start the API with:")
            st.code("uv run uvicorn src.api.main:app --port 8000")
            return False
        
        st.markdown("---")
        
        # Use cases
        st.subheader("📋 Use Cases")
        use_cases_data = api_get_use_cases()
        
        if use_cases_data:
            # Group by category
            categories = {"consumer": "📱 Consumer", "business": "🏢 Business", "operator": "📡 Operator"}
            
            for cat_key, cat_name in categories.items():
                with st.expander(cat_name):
                    for uc in use_cases_data["use_cases"]:
                        if uc["category"] == cat_key:
                            st.caption(f"**{uc['use_case_id']}**: {uc['name']}")
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        max_iterations = st.slider("Max Iterations", 1, 5, 3)
        
        st.markdown("---")
        
        # Lessons management
        st.subheader("📚 Lessons")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View", use_container_width=True):
                st.session_state.show_lessons = True
        with col2:
            if st.button("Clear", use_container_width=True):
                if api_clear_lessons():
                    st.success("Cleared!")
                    time.sleep(1)
                    st.rerun()
        
        return max_iterations


def render_lessons_modal():
    """Render lessons in an expander."""
    if st.session_state.get("show_lessons"):
        with st.expander("📚 Learned Design Lessons", expanded=True):
            lessons_data = api_get_lessons()
            if lessons_data and lessons_data["use_cases"]:
                for uc in lessons_data["use_cases"]:
                    st.markdown(f"**{uc['use_case_id']}: {uc['use_case_name']}**")
                    for lesson in uc["lessons"]:
                        st.markdown(f"- {lesson['lesson']}")
                    st.markdown("---")
            else:
                st.info("No lessons learned yet. Run a design workflow to generate lessons.")
            
            if st.button("Close"):
                st.session_state.show_lessons = False
                st.rerun()


def render_result(result: dict):
    """Render the design result."""
    
    # Header
    if result.get("is_approved"):
        st.success(f"✅ Design APPROVED - Iteration {result.get('iteration', 0)}")
    else:
        st.warning(f"⚠️ Design completed - Iteration {result.get('iteration', 0)}")
    
    # Use case info
    st.markdown(f"### 📌 {result.get('use_case_id')}: {result.get('use_case_name')}")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📐 Design", "📋 Requirements", "💬 Feedback", "📝 Summary"])
    
    with tab1:
        st.markdown("## System Design")
        st.markdown(result.get("system_design", "No design generated"))
    
    with tab2:
        st.markdown("## Requirements")
        st.markdown(result.get("requirement_list", "No requirements found"))
    
    with tab3:
        st.markdown("## Feedback Evaluation")
        st.markdown(result.get("feedback", "No feedback"))
    
    with tab4:
        st.markdown("## Description Summary")
        st.markdown(result.get("description_summary", "No summary"))


def poll_for_result(task_id: str, placeholder):
    """Poll for task completion and show progress."""
    
    status_messages = {
        "pending": "⏳ Starting workflow...",
        "running": "🔄 Running design workflow...",
        "completed": "✅ Completed!",
        "failed": "❌ Failed"
    }
    
    step_icons = {
        "query_analysis": "🔍 Query Analysis",
        "design": "📐 Design Generation", 
        "feedback": "💬 Feedback Evaluation",
        "done": "✅ Done"
    }
    
    max_polls = 120  # 2 minutes max
    poll_count = 0
    
    while poll_count < max_polls:
        status = api_get_design_status(task_id)
        
        if not status:
            placeholder.error("Failed to get task status")
            return None
        
        task_status = status.get("status")
        
        # Show progress
        with placeholder.container():
            st.markdown(f"### {status_messages.get(task_status, task_status)}")
            
            progress = status.get("progress")
            if progress:
                current_step = progress.get("current_step", "")
                iteration = progress.get("iteration", 0)
                max_iter = progress.get("max_iterations", 3)
                
                st.markdown(f"**Step:** {step_icons.get(current_step, current_step)}")
                st.markdown(f"**Iteration:** {iteration}/{max_iter}")
                
                # Progress bar
                steps = ["query_analysis", "design", "feedback"]
                if current_step in steps:
                    progress_pct = (steps.index(current_step) + 1) / len(steps)
                    st.progress(progress_pct)
        
        # Check if done
        if task_status == "completed":
            return status.get("result")
        elif task_status == "failed":
            placeholder.error(f"Task failed: {status.get('error')}")
            return None
        
        time.sleep(1)
        poll_count += 1
    
    placeholder.error("Timeout waiting for result")
    return None


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main Streamlit app."""
    
    # Initialize session state
    if "show_lessons" not in st.session_state:
        st.session_state.show_lessons = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    
    # Sidebar
    max_iterations = render_sidebar()
    if max_iterations is False:  # API offline
        return
    
    # Main content
    st.title("🤖 ETSI Use Case Design System")
    st.markdown("Design AI agentic systems based on ETSI GR ENI 055 use cases")
    
    # Lessons modal
    render_lessons_modal()
    
    st.markdown("---")
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your design query",
            placeholder="e.g., Design a system for the smart life use case",
            label_visibility="collapsed"
        )
    
    with col2:
        run_button = st.button("🚀 Design", type="primary", use_container_width=True)
    
    # Example queries
    st.caption("**Examples:** Design a system for smart life | Create an AI system for disaster handling | Build a system for game acceleration")
    
    st.markdown("---")
    
    # Run workflow
    if run_button and query:
        # Start the task
        result = api_start_design(query, max_iterations)
        
        if result and "task_id" in result:
            task_id = result["task_id"]
            
            # Create placeholder for progress
            progress_placeholder = st.empty()
            
            # Poll for result
            final_result = poll_for_result(task_id, progress_placeholder)
            
            if final_result:
                st.session_state.last_result = final_result
                progress_placeholder.empty()
        else:
            st.error(f"Failed to start design: {result}")
    
    # Show last result
    if st.session_state.last_result:
        render_result(st.session_state.last_result)


if __name__ == "__main__":
    main()