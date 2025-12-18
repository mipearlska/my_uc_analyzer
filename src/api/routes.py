"""
FastAPI routes for the ETSI Design System.
"""

import uuid
from datetime import datetime
from typing import Dict
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.api.models import (
    TaskStatus,
    WorkflowStep,
    DesignRequest,
    DesignTaskResponse,
    DesignStatusResponse,
    DesignResult,
    WorkflowProgress,
    UseCaseInfo,
    UseCaseListResponse,
    UseCaseLessonsResponse,
    AllLessonsResponse,
    LessonInfo,
    HealthResponse,
)
from src.vector_store import ChromaVectorStore
from src.agents.memory import LongTermMemory
from src.agents.graph import run_design_workflow
from src.etsi_parser import parse_etsi_document


# =============================================================================
# Global State (in production, use Redis or database)
# =============================================================================

# Task storage
tasks: Dict[str, dict] = {}

# Shared resources (initialized on startup)
vector_store: ChromaVectorStore = None
memory: LongTermMemory = None


# Use case definitions
USE_CASES = {
    "5.1.1": ("AI Agents to Enable Smart Life", "consumer"),
    "5.1.2": ("Network-Assisted Collaborative Robots", "consumer"),
    "5.1.3": ("AI Phone", "consumer"),
    "5.2.1": ("AI Agent-based Customized Network for Smart City Traffic Monitoring", "business"),
    "5.2.2": ("AI Agents-Based Customized Network for Smart Construction Sites", "business"),
    "5.2.3": ("AI Agent Ensuring Game Acceleration Experience", "business"),
    "5.2.4": ("AI Agent-Assisted Collaborative Energy Distribution in Power Enterprises", "business"),
    "5.3.1": ("AI Agent-Based Autonomous Network Management", "operator"),
    "5.3.2": ("AI Agent-Based Disaster Handling Network Management", "operator"),
    "5.3.3": ("AI Agent-Based Time-Sensitive Network Management", "operator"),
    "5.3.4": ("AI Agent-Driven Core Network Signalling Optimization", "operator"),
    "5.3.5": ("AI Agent-Based Core Networks to Enhance User Experience", "operator"),
}


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api", tags=["ETSI Design System"])


# =============================================================================
# Initialization
# =============================================================================

def initialize_resources():
    """Initialize vector store and memory."""
    global vector_store, memory
    
    PDF_PATH = Path("data/ETSI_AI-Agent_core_usecases.pdf")
    
    # Initialize vector store
    vector_store = ChromaVectorStore()
    
    
    # Initialize memory
    memory = LongTermMemory()
    
    return vector_store, memory


# =============================================================================
# Background Task
# =============================================================================

def run_design_task(task_id: str, query: str, max_iterations: int):
    """Run the design workflow as a background task."""
    global vector_store, memory, tasks
    
    try:
        # Update status to running
        tasks[task_id]["status"] = TaskStatus.RUNNING
        tasks[task_id]["progress"] = {
            "current_step": WorkflowStep.QUERY_ANALYSIS,
            "iteration": 0,
            "max_iterations": max_iterations,
            "steps_completed": []
        }
        
        # Run the workflow
        result = run_design_workflow(
            user_query=query,
            vector_store=vector_store,
            memory=memory,
            max_iterations=max_iterations,
            verbose=False  # Disable console output
        )
        
        # Update task with result
        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["completed_at"] = datetime.now()
        tasks[task_id]["result"] = {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED,
            "use_case_id": result.get("use_case_id"),
            "use_case_name": result.get("use_case_name"),
            "description_summary": result.get("description_summary"),
            "requirement_list": result.get("requirement_list"),
            "system_design": result.get("system_design"),
            "feedback": result.get("feedback"),
            "is_approved": result.get("is_approved", False),
            "iteration": result.get("iteration", 0),
            "final_response": result.get("final_response"),
            "error": result.get("error"),
            "created_at": tasks[task_id]["created_at"],
            "completed_at": tasks[task_id]["completed_at"],
        }
        
    except Exception as e:
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = datetime.now()


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global vector_store, memory
    
    if vector_store is None:
        initialize_resources()
    
    # Count lessons
    lesson_count = 0
    for uc_id, (name, _) in USE_CASES.items():
        lessons = memory.get_lessons(uc_id, name)
        lesson_count += len(lessons.lessons)
    
    return HealthResponse(
        status="healthy",
        vector_store_count=vector_store.count() if vector_store else 0,
        lessons_count=lesson_count
    )


@router.get("/use-cases", response_model=UseCaseListResponse)
async def list_use_cases():
    """List all available use cases."""
    use_cases = [
        UseCaseInfo(
            use_case_id=uc_id,
            name=name,
            category=category
        )
        for uc_id, (name, category) in USE_CASES.items()
    ]
    
    return UseCaseListResponse(
        use_cases=use_cases,
        total=len(use_cases)
    )


@router.post("/design", response_model=DesignTaskResponse)
async def start_design(request: DesignRequest, background_tasks: BackgroundTasks):
    """Start a new design workflow."""
    global vector_store, memory
    
    # Initialize resources if needed
    if vector_store is None:
        initialize_resources()
    
    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "query": request.query,
        "max_iterations": request.max_iterations,
        "created_at": datetime.now(),
        "completed_at": None,
        "progress": None,
        "result": None,
        "error": None,
    }
    
    # Start background task
    background_tasks.add_task(
        run_design_task,
        task_id,
        request.query,
        request.max_iterations
    )
    
    return DesignTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Design workflow started"
    )


@router.get("/design/{task_id}", response_model=DesignStatusResponse)
async def get_design_status(task_id: str):
    """Get the status of a design task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return DesignStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        result=task.get("result"),
        error=task.get("error")
    )


@router.get("/lessons", response_model=AllLessonsResponse)
async def get_all_lessons():
    """Get all learned lessons."""
    global memory
    
    if memory is None:
        initialize_resources()
    
    use_cases = []
    total_lessons = 0
    
    for uc_id, (name, _) in USE_CASES.items():
        lessons_data = memory.get_lessons(uc_id, name)
        if lessons_data.lessons:
            use_cases.append(UseCaseLessonsResponse(
                use_case_id=uc_id,
                use_case_name=name,
                lessons=[
                    LessonInfo(
                        lesson=l.lesson,
                        created_at=l.created_at
                    )
                    for l in lessons_data.lessons
                ]
            ))
            total_lessons += len(lessons_data.lessons)
    
    return AllLessonsResponse(
        use_cases=use_cases,
        total_lessons=total_lessons
    )


@router.get("/lessons/{use_case_id}", response_model=UseCaseLessonsResponse)
async def get_use_case_lessons(use_case_id: str):
    """Get lessons for a specific use case."""
    global memory
    
    if memory is None:
        initialize_resources()
    
    if use_case_id not in USE_CASES:
        raise HTTPException(status_code=404, detail="Use case not found")
    
    name, _ = USE_CASES[use_case_id]
    lessons_data = memory.get_lessons(use_case_id, name)
    
    return UseCaseLessonsResponse(
        use_case_id=use_case_id,
        use_case_name=name,
        lessons=[
            LessonInfo(
                lesson=l.lesson,
                created_at=l.created_at
            )
            for l in lessons_data.lessons
        ]
    )


@router.delete("/lessons")
async def clear_all_lessons():
    """Clear all learned lessons."""
    global memory
    
    lessons_path = Path("./data/test_lessons.json")
    if lessons_path.exists():
        lessons_path.unlink()
    
    memory = LongTermMemory("./data/test_lessons.json")
    
    return {"message": "All lessons cleared"}