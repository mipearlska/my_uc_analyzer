"""
FastAPI application for ETSI Design System.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, initialize_resources


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    print("🚀 Starting ETSI Design System API...")
    initialize_resources()
    print("✓ Resources initialized")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="ETSI Use Case Design System API",
    description="""
    API for designing AI agentic systems based on ETSI use cases.
    
    Features:
    - Query Analysis Agent (Ollama) - identifies use cases
    - Design Agent with ReAct + MCP (Groq) - researches and designs  
    - Feedback Agent (Groq) - evaluates designs
    - Long-term memory for learned lessons
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ETSI Use Case Design System API",
        "docs": "/docs",
        "health": "/api/health"
    }