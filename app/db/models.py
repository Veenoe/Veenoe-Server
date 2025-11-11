# app/db/models.py

from pydantic import BaseModel, Field
from pydantic_core import core_schema  # Correct import (unchanged)
from typing import List, Optional, Any
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """
        Custom validation for MongoDB ObjectId in Pydantic V2.
        Accepts ObjectId instance or valid string, serializes to str.
        """
        
        # Validator function (unchanged)
        def validate_from_str(v: str) -> ObjectId:
            if not ObjectId.is_valid(v):
                raise ValueError("Invalid ObjectId")
            return ObjectId(v)

        return core_schema.union_schema(
            [
                # Schema 1: Check if it's already an ObjectId instance
                core_schema.is_instance_schema(ObjectId),
                
                # Schema 2: If not, try to validate it from a string
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        # FIXED: Use no_info_plain_validator_function
                        core_schema.no_info_plain_validator_function(validate_from_str),
                    ]
                ),
            ],
            # Serialization: Convert ObjectId to str
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


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
    ai_evaluation: Optional[LLMEvaluation] = None  # Stores structured LLM output


class VivaSession(BaseModel):
    """
    The main document model for a single viva session.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    student_name: str
    topic: str
    class_level: str
    status: str = Field(default="active")  # e.g., "active", "completed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    transcript: List[Message] = Field(default_factory=list)

    # Pydantic V2 Config (unchanged from previous fix)
    model_config = {
        "populate_by_name": True,  # Replaces 'allow_population_by_field_name'
        "arbitrary_types_allowed": True,  # Required for PyObjectId to work
        "json_encoders": {ObjectId: str}  # Handles ObjectId -> str serialization
    }