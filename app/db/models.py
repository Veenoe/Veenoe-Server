# app/db/models.py

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class LLMEvaluation(BaseModel):
    """
    A structured response from the LLM, containing both
    evaluation of the previous answer and the next question.
    """
    evaluation: str = Field(description="AI's feedback on the student's last answer.")
    new_question: str = Field(description="AI's next question for the student.")

class Message(BaseModel):
    """A single message in the viva transcript."""
    speaker: str = Field(description="'ai' or 'user'")
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ai_evaluation: Optional[LLMEvaluation] = None # Stores structured LLM output

class VivaSession(BaseModel):
    """
    The main document model for a single viva session.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    student_name: str  # Changed from user_id
    topic: str
    class_level: str
    status: str = Field(default="active") # e.g., "active", "completed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    transcript: List[Message] = Field(default_factory=list)

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}