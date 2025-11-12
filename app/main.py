# app/main.py

from fastapi import (
    FastAPI, 
    WebSocket, 
    WebSocketDisconnect, 
    Depends,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import asyncio
import json
from contextlib import asynccontextmanager

# FIXED: Import functions separately + the db instance
from app.db.database import db as mongo_db, connect_to_database, close_database_connection  # <-- ADD FUNCTIONS
from app.services.session_service import SessionService, get_session_service
from app.services.orchestrator import VivaOrchestrator
from app.db.models import VivaSession

# --- Database Lifecycle (UNCHANGED) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events.
    Connects to MongoDB on startup, disconnects on shutdown.
    """
    print("--- Main: Connecting to database... ---")
    await connect_to_database()  # <-- FIXED: Call function directly
    yield
    print("--- Main: Disconnecting from database... ---")
    await close_database_connection()  # <-- FIXED: Call function directly

# --- Application Setup (UNCHANGED) ---

app = FastAPI(
    title="Viva AI Backend",
    description="Manages real-time, streaming AI viva sessions.",
    version="1.0.0",
    lifespan=lifespan
)

# --- 2. ADD THE CORS MIDDLEWARE ---
# This is the standard, secure way to handle this in production.
# We explicitly list the origins that are allowed to talk to our API.

origins = [
    "http://localhost:3000", # The origin for your Next.js dev server
    # "https://your-production-domain.com", # TODO: Add your production frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- HTTP Endpoints (UNCHANGED) ---

class VivaStartRequest(BaseModel):
    """The request body for starting a new viva."""
    student_name: str
    topic: str
    class_level: str = "10th Grade"

class VivaStartResponse(BaseModel):
    """The response containing the new session ID."""
    session_id: str

@app.post("/start-viva", response_model=VivaStartResponse)
async def start_viva(
    request: VivaStartRequest,
    service: SessionService = Depends(get_session_service)
):
    """
    HTTP endpoint to create a new viva session.
    This is a fast, stateless request. It just creates the DB entry.
    """
    session_id = await service.create_new_viva_session(
        student_name=request.student_name,
        topic=request.topic,
        class_level=request.class_level,
    )
    return VivaStartResponse(session_id=session_id)

@app.get("/viva-history", response_model=List[VivaSession])
async def get_history(
    student_name: str,
    service: SessionService = Depends(get_session_service)
):
    """
    HTTP endpoint to get all past viva sessions for a specific student.
    """
    return await service.get_viva_history_for_user(student_name)

# --- WebSocket Endpoint (UNCHANGED) ---

@app.websocket("/ws/viva/{session_id}")
async def websocket_viva_endpoint(
    websocket: WebSocket, 
    session_id: str
):
    """
    This is the main, stateful WebSocket endpoint for conducting the viva.
    """
    await websocket.accept()
    
    # 1. Initialize the Orchestrator for this specific connection
    orchestrator = VivaOrchestrator(websocket, session_id)
    
    # 2. Connect to DB, find session, and connect to ASR
    if not await orchestrator.initialize():
        # If session_id is invalid or DB fails, close the connection
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=f"Session {session_id} not found or invalid."
        )
        return

    try:
        # 3. Get the first question and stream it to the client
        await orchestrator.start_viva()

        # 4. Enter the main loop: listen for audio from the client
        while True:
            data = await websocket.receive()
            
            if isinstance(data, bytes):
                print(f"--- Main: Received {len(data)} audio bytes ---")
                await orchestrator.handle_audio_chunk(data)
            
            elif isinstance(data, str):
                # Handle text-based control messages if needed
                print(f"--- Main: Received text message: {data} ---")
                pass

    except WebSocketDisconnect:
        print(f"--- Main: Client disconnected from session {session_id} ---")
    
    except Exception as e:
        print(f"--- Main: An error occurred in session {session_id}: {e} ---")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    
    finally:
        # 5. CRITICAL: Clean up all connections
        await orchestrator.disconnect()