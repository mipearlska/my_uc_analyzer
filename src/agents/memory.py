"""
Long-term memory for learned design lessons.

Persists across conversations - stores lessons learned
from designing systems for each use case.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DesignLesson(BaseModel):
    """A single lesson learned from a design iteration."""
    lesson: str
    created_at: datetime = Field(default_factory=datetime.now)


class UseCaseLessons(BaseModel):
    """All lessons for a specific use case."""
    use_case_id: str
    use_case_name: str
    lessons: list[DesignLesson] = Field(default_factory=list)
    
    def add_lesson(self, lesson: str) -> None:
        """Add a new lesson."""
        self.lessons.append(DesignLesson(lesson=lesson))
    
    def get_lessons_text(self) -> str:
        """Get all lessons as formatted text."""
        if not self.lessons:
            return "No previous lessons for this use case."
        
        lines = [f"Learned lessons for {self.use_case_name}:"]
        for i, lesson in enumerate(self.lessons, 1):
            lines.append(f"  {i}. {lesson.lesson}")
        return "\n".join(lines)


class LongTermMemory:
    """
    Persistent storage for design lessons.
    
    Saves to JSON file so lessons persist across sessions.
    
    Usage:
        memory = LongTermMemory("./data/lessons.json")
        
        # Get lessons for a use case
        lessons = memory.get_lessons("5.1.1", "AI Agents to Enable Smart Life")
        
        # Add a new lesson
        memory.add_lesson("5.1.1", "AI Agents to Enable Smart Life", 
                         "Privacy mechanisms must be included")
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    MEMORY_PATH = BASE_DIR.parent/"data"/"test_lessons.json"

    def __init__(self, storage_path: str | Path = MEMORY_PATH):
        self.storage_path = Path(storage_path)
        self._data: dict[str, UseCaseLessons] = {}
        self._load()
    
    def _load(self) -> None:
        """Load lessons from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    raw_data = json.load(f)
                    for use_case_id, lesson_data in raw_data.items():
                        self._data[use_case_id] = UseCaseLessons(**lesson_data)
            except Exception as e:
                print(f"Warning: Could not load lessons: {e}")
                self._data = {}
        else:
            self._data = {}
    
    def _save(self) -> None:
        """Save lessons to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        raw_data = {}
        for use_case_id, lessons in self._data.items():
            raw_data[use_case_id] = {
                "use_case_id": lessons.use_case_id,
                "use_case_name": lessons.use_case_name,
                "lessons": [
                    {
                        "lesson": l.lesson,
                        "created_at": l.created_at.isoformat()
                    }
                    for l in lessons.lessons
                ]
            }
        
        with open(self.storage_path, "w") as f:
            json.dump(raw_data, f, indent=2)
    
    def get_lessons(self, use_case_id: str, use_case_name: str) -> UseCaseLessons:
        """
        Get lessons for a use case.
        
        Creates empty entry if use case not seen before.
        """
        if use_case_id not in self._data:
            self._data[use_case_id] = UseCaseLessons(
                use_case_id=use_case_id,
                use_case_name=use_case_name
            )
        return self._data[use_case_id]
    
    def add_lesson(self, use_case_id: str, use_case_name: str, lesson: str) -> None:
        """Add a lesson for a use case and save."""
        lessons = self.get_lessons(use_case_id, use_case_name)
        lessons.add_lesson(lesson)
        self._save()
    
    def get_all_lessons_summary(self) -> str:
        """Get summary of all lessons for all use cases."""
        if not self._data:
            return "No design lessons stored yet."
        
        lines = ["=== Learned Design Lessons ===\n"]
        for use_case_id, lessons in sorted(self._data.items()):
            lines.append(f"\n{use_case_id}: {lessons.use_case_name}")
            lines.append("-" * 40)
            if lessons.lessons:
                for i, lesson in enumerate(lessons.lessons, 1):
                    lines.append(f"  {i}. {lesson.lesson}")
            else:
                lines.append("  (no lessons yet)")
        
        return "\n".join(lines)