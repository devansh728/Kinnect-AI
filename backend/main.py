# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.audio_handler import AudioHandler
from backend.websocket.handlers import router as ws_router
from backend.websocket.connection_manager import connection_manager
import chromadb
import os

# Define lifespan event to load heavy models once
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the AudioHandler (which loads Whisper base model)
    print("🚀 Starting Kinnect AI Backend Server...")
    print("Loading models (Whisper) - this might take a moment...")
    
    # Optional config for whisper model size
    whisper_model_size = os.getenv("WHISPER_MODEL", "base")
    app.state.audio_handler = AudioHandler(whisper_model=whisper_model_size)
    
    # Initialize ChromaDB client to ensure directory exists
    app.state.db_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Initialize Scheduler if needed (we'll import scheduler here once it's created)
    try:
        from backend.scheduler import call_scheduler
        await call_scheduler.start()
        print("⏰ Call Scheduler started successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Call Scheduler failed to start: {e}")
        
    print("✅ Kinnect AI Server is ready.")
    
    yield
    
    # Shutdown: Clean up scheduler
    print("🛑 Shutting down server...")
    try:
        from backend.scheduler import call_scheduler
        await call_scheduler.shutdown()
        print("⏰ Call Scheduler shutdown.")
    except Exception as e:
        print(f"⚠️ Warning: Scheduler shutdown error: {e}")
        
    print("👋 Goodbye.")

app = FastAPI(
    title="Kinnect AI Backend",
    description="Backend API and WebSocket check-in system for elderly cognitive care.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Open by default, configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include WebSocket router
app.include_router(ws_router)

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat() if "datetime" in globals() else None
    }

@app.get("/status")
async def get_status():
    """Returns system status including collection size and active connection counts."""
    # Count of active WebSocket connections
    connections_count = sum(len(ws_list) for ws_list in connection_manager.active_connections.values())
    active_users = list(connection_manager.active_connections.keys())
    
    # Check ChromaDB collection size
    try:
        db = chromadb.PersistentClient(path="./chroma_db")
        collection = db.get_or_create_collection(name="patient_memories")
        memories_count = collection.count()
    except Exception as e:
        memories_count = f"Error: {str(e)}"
        
    return {
        "active_connections": connections_count,
        "connected_users": active_users,
        "chromadb_memories_count": memories_count,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

# Fix import for datetime inside endpoint
from datetime import datetime
