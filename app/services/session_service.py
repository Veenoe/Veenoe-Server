# app/services/session_service.py

from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List
from bson import ObjectId

from app.db.database import get_db
from app.db.models import VivaSession
# from app.auth import User # <-- REMOVED

class SessionService:
    """
    Handles the business logic for creating and retrieving viva sessions.
    This is a stateless service.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = self.db["viva_sessions"]

    async def create_new_viva_session(
        self, 
        student_name: str, # <-- ADDED
        topic: str, 
        class_level: str, 
        # user: User # <-- REMOVED
    ) -> str:
        """
        Creates a new, empty viva session in the database.
        """
        print(f"--- Session Service: Creating new session for {student_name} on '{topic}' ---")
        
        # 1. Create the session object
        new_session = VivaSession(
            student_name=student_name, # <-- MODIFIED
            topic=topic,
            class_level=class_level,
            status="active",
            transcript=[]
        )
        
        # 2. Insert into MongoDB
        result = await self.collection.insert_one(
            new_session.model_dump(by_alias=True, exclude={'id'})
        )
        
        if not result.inserted_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create new session in database."
            )
        
        # 3. Return the new session ID as a string
        return str(result.inserted_id)

    async def get_viva_history_for_user(self, student_name: str) -> List[VivaSession]: # <-- MODIFIED
        """
        Retrieves all past viva sessions for a given student.
        """
        print(f"--- Session Service: Fetching history for {student_name} ---") # <-- MODIFIED
        
        sessions = []
        # Find sessions matching the student_name
        cursor = self.collection.find({"student_name": student_name}) # <-- MODIFIED
        
        async for session_data in cursor:
            sessions.append(VivaSession.model_validate(session_data))
            
        return sessions

# --- FastAPI Dependency ---

async def get_session_service() -> SessionService:
    """
    FastAPI dependency to inject a SessionService instance
    with a database connection.
    """
    db = await get_db()
    return SessionService(db)