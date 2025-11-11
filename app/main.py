# app/main.py

from fastapi import (
    FastAPI, 
    WebSocket, 
    WebSocketDisconnect, 
    Depends,
    status
)
from pydantic import BaseModel
from typing import List
import asyncio
import json
from contextlib import asynccontextmanager # 1. IMPORT THIS

from app.db.database import db as mongo_db
from app.services.session_service import SessionService, get_session_service
from app.services.orchestrator import VivaOrchestrator
# from app.auth import User, get_current_user # <-- 4. REMOVED
from app.db.models import VivaSession

# --- 1. Database Lifecycle (NEW LIFESPAN) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events.
    Connects to MongoDB on startup, disconnects on shutdown.
    """
    print("--- Main: Connecting to database... ---")
    await mongo_db.connect_to_database()
    yield
    print("--- Main: Disconnecting from database... ---")
    await mongo_db.close_database_connection()

# --- 2. Application Setup ---

app = FastAPI(
    title="Viva AI Backend",
    description="Manages real-time, streaming AI viva sessions.",
    version="1.0.0",
    lifespan=lifespan  # 2. ADD THIS
)

# --- 4. HTTP Endpoints ---

class VivaStartRequest(BaseModel):
    """The request body for starting a new viva."""
    student_name: str  # 4. ADDED
    topic: str
    class_level: str = "10th Grade"

class VivaStartResponse(BaseModel):
    """The response containing the new session ID."""
    session_id: str

@app.post("/start-viva", response_model=VivaStartResponse)
async def start_viva(
    request: VivaStartRequest,
    # user: User = Depends(get_current_user), # <-- 4. REMOVED
    service: SessionService = Depends(get_session_service)
):
    """
    HTTP endpoint to create a new viva session.
    This is a fast, stateless request. It just creates the DB entry.
    """
    session_id = await service.create_new_viva_session(
        student_name=request.student_name, # 4. ADDED
        topic=request.topic,
        class_level=request.class_level,
        # user=user # <-- 4. REMOVED
    )
    return VivaStartResponse(session_id=session_id)

@app.get("/viva-history", response_model=List[VivaSession])
async def get_history(
    student_name: str, # 4. ADDED (as a query param)
    # user: User = Depends(get_current_user), # <-- 4. REMOVED
    service: SessionService = Depends(get_session_service)
):
    """
    HTTP endpoint to get all past viva sessions for a specific student.
    """
    return await service.get_viva_history_for_user(student_name) # 4. MODIFIED

# --- 5. The Main WebSocket Endpoint ---

@app.websocket("/ws/viva/{session_id}")
async def websocket_viva_endpoint(
    websocket: WebSocket, 
    session_id: str
    # No auth changes needed here as it was already skipped
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