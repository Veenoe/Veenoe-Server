import asyncio
from fastapi import WebSocket
from bson import ObjectId

from app.db.database import get_db
from app.db.models import VivaSession, Message, LLMEvaluation
from app.services.sarvam_asr_service import SarvamASRService
from app.services.gemini_llm_service import get_ai_evaluation, get_ai_first_question
from app.services.sarvam_tts_service import text_to_audio_stream

class VivaOrchestrator:
    """
    Manages the entire real-time viva for a single WebSocket connection.

    1. Initializes all services (ASR, LLM, TTS).
    2. Handles the bi-directional audio stream with the client.
    3. Orchestrates the flow: User Audio -> ASR -> LLM -> TTS -> Client Audio.
    4. Updates the database with the transcript in real-time.
    """

    def __init__(self, client_ws: WebSocket, session_id: str):
        self.client_ws = client_ws
        self.session_id = session_id
        self.db = None
        self.session_collection = None
        self.viva_session: VivaSession | None = None

        # Initialize our stateful ASR service
        self.asr_service = SarvamASRService(on_transcript=self.on_transcript_received)
        self.is_processing_llm = False  # A "lock" to prevent concurrent LLM calls

    async def initialize(self):
        """
        Connects to the database and loads the viva session.
        Returns True on success, False on failure.
        """
        try:
            self.db = await get_db()
            self.session_collection = self.db["viva_sessions"]

            if self.session_collection is None:
                raise RuntimeError("MongoDB collection not initialized.")

            # Fetch the session from MongoDB
            session_data = await self.session_collection.find_one({"_id": ObjectId(self.session_id)})

            if not session_data:
                print(f"--- Orchestrator: Error! Session {self.session_id} not found. ---")
                return False

            self.viva_session = VivaSession.model_validate(session_data)

            # Connect ASR service
            await self.asr_service.connect()

            return True

        except Exception as e:
            print(f"--- Orchestrator: Initialization failed! {e} ---")
            return False

    async def start_viva(self):
        """
        Starts the viva by getting and sending the first AI question.
        """
        if not self.viva_session:
            return

        if len(self.viva_session.transcript) > 0:
            print("--- Orchestrator: Resuming viva. Waiting for user audio. ---")
            return

        print(f"--- Orchestrator: Starting new viva for topic: {self.viva_session.topic} ---")

        # 1. Get the first question from the LLM
        llm_response = await get_ai_first_question(
            topic=self.viva_session.topic,
            class_level=self.viva_session.class_level
        )

        # 2. Save the first AI message to DB
        ai_message = await self.save_message_to_db(
            speaker="ai",
            text=llm_response.new_question,
            evaluation=llm_response
        )

        # 3. Stream the question audio to the client
        await self.stream_ai_audio_to_client(ai_message)

    async def handle_audio_chunk(self, audio_chunk: bytes):
        """
        Receives raw audio chunks from the client's WebSocket.
        """
        if self.asr_service and not self.is_processing_llm:
            await self.asr_service.send_audio_chunk(audio_chunk)

    async def on_transcript_received(self, transcript: str):
        """
        Callback triggered by ASR service when a final transcript is ready.
        """
        if not self.viva_session:
            return

        try:
            self.is_processing_llm = True

            # 1. Save user's transcript
            await self.save_message_to_db(speaker="user", text=transcript)

            # 2. Get AI evaluation
            llm_response = await get_ai_evaluation(
                topic=self.viva_session.topic,
                class_level=self.viva_session.class_level,
                history=self.viva_session.transcript,
                student_answer=transcript
            )

            # 3. Save AI response
            ai_message = await self.save_message_to_db(
                speaker="ai",
                text=f"{llm_response.evaluation} {llm_response.new_question}",
                evaluation=llm_response
            )

            # 4. Stream AI response as audio
            await self.stream_ai_audio_to_client(ai_message)

        except Exception as e:
            print(f"--- Orchestrator: Error in viva loop! {e} ---")

        finally:
            self.is_processing_llm = False

    async def stream_ai_audio_to_client(self, ai_message: Message):
        """
        Streams AI's audio response to the client via WebSocket.
        """
        print(f"--- Orchestrator: Streaming AI audio to client... ---")

        audio_stream = text_to_audio_stream(text=ai_message.text)

        async for audio_chunk in audio_stream:
            await self.client_ws.send_bytes(audio_chunk)

        await self.client_ws.send_json({"type": "speech_end"})
        print("--- Orchestrator: Finished streaming AI audio. ---")

    async def save_message_to_db(
        self,
        speaker: str,
        text: str,
        evaluation: LLMEvaluation = None
    ) -> Message:
        """
        Creates and saves a Message to MongoDB and local transcript.
        """
        if self.session_collection is None or self.viva_session is None:
            raise Exception("Database or session not initialized")

        message = Message(
            speaker=speaker,
            text=text,
            ai_evaluation=evaluation
        )

        self.viva_session.transcript.append(message)

        await self.session_collection.update_one(
            {"_id": ObjectId(self.session_id)},
            {"$push": {"transcript": message.model_dump()}}
        )

        print(f"--- Orchestrator: Saved message to DB: '{text[:30]}...' ---")
        return message

    async def disconnect(self):
        """
        Cleans up all services and marks the session as completed.
        """
        if self.asr_service:
            await self.asr_service.close()

        # ✅ FIX: Motor collection cannot be truth-tested
        if self.session_collection is not None:
            await self.session_collection.update_one(
                {"_id": ObjectId(self.session_id)},
                {"$set": {"status": "completed"}}
            )

        print(f"--- Orchestrator: Session {self.session_id} disconnected. ---")
