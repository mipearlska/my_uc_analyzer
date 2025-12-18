"""
Pydantic models for FastAPI endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class TaskStatus(str, Enum):
    """Status of a design task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStep(str, Enum):
    """Current step in the workflow."""
    QUERY_ANALYSIS = "query_analysis"
    DESIGN = "design"
    FEEDBACK = "feedback"
    DONE = "done"


# =============================================================================
# Request Models
# =============================================================================

class DesignRequest(BaseModel):
    """Request to start a design workflow."""
    query: str = Field(
        ...,
        description="User query describing what to design",
        min_length=5,
        examples=["Design a system for the smart life use case"]
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum feedback loop iterations",
        ge=1,
        le=5
    )


# =============================================================================
# Response Models
# =============================================================================

class UseCaseInfo(BaseModel):
    """Information about a use case."""
    use_case_id: str
    name: str
    category: str


class UseCaseListResponse(BaseModel):
    """List of available use cases."""
    use_cases: list[UseCaseInfo]
    total: int


class DesignTaskResponse(BaseModel):
    """Response when starting a design task."""
    task_id: str
    status: TaskStatus
    message: str


class WorkflowProgress(BaseModel):
    """Progress information for a running workflow."""
    current_step: WorkflowStep
    iteration: int
    max_iterations: int
    steps_completed: list[str]


class DesignResult(BaseModel):
    """Result of a completed design workflow."""
    task_id: str
    status: TaskStatus
    use_case_id: Optional[str] = None
    use_case_name: Optional[str] = None
    description_summary: Optional[str] = None
    requirement_list: Optional[str] = None
    system_design: Optional[str] = None
    feedback: Optional[str] = None
    is_approved: bool = False
    iteration: int = 0
    final_response: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class DesignStatusResponse(BaseModel):
    """Response for design task status."""
    task_id: str
    status: TaskStatus
    progress: Optional[WorkflowProgress] = None
    result: Optional[DesignResult] = None
    error: Optional[str] = None


class LessonInfo(BaseModel):
    """A single design lesson."""
    lesson: str
    created_at: datetime


class UseCaseLessonsResponse(BaseModel):
    """Lessons for a specific use case."""
    use_case_id: str
    use_case_name: str
    lessons: list[LessonInfo]


class AllLessonsResponse(BaseModel):
    """All lessons across all use cases."""
    use_cases: list[UseCaseLessonsResponse]
    total_lessons: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_store_count: int
    lessons_count: int