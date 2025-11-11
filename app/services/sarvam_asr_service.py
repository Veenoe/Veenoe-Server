# app/services/sarvam_asr_service.py

import asyncio
import base64
from sarvamai import AsyncSarvamAI
from app.config import settings
from typing import Callable, Awaitable

TranscriptCallback = Callable[[str], Awaitable[None]]

class SarvamASRService:
    def __init__(self, on_transcript: TranscriptCallback):
        self.client = AsyncSarvamAI(api_subscription_key=settings.SARVAM_API_KEY)
        self.on_transcript = on_transcript
        self.ws = None
        self._listener_task = None
        self._is_connected = False
        
        # We will configure these in the connect() call
        self.input_audio_codec = "pcm" # We are sending raw PCM (L16)
        self.sample_rate = 16000 # Must match the client

    async def connect(self, language_code: str = "en-IN"):
        try:
            # We configure the connection based on the docs
            self.ws = await self.client.speech_to_text_streaming.connect(
                language_code=language_code,
                model="saarika:v2.5",
                high_vad_sensitivity=True,
                vad_signals=True,
                # --- Add these based on docs ---
                sample_rate=self.sample_rate,
                input_audio_codec=self.input_audio_codec
            )
            self._is_connected = True
            print(f"--- ASR Service: Connected to Sarvam (codec: {self.input_audio_codec}, rate: {self.sample_rate}) ---")
            
            self._listener_task = asyncio.create_task(self._listen())
        except Exception as e:
            print(f"--- ASR Service: Connection failed! {e} ---")
            self._is_connected = False
            raise

    async def _listen(self):
        """
        A private, long-running task that listens for messages
        from the Sarvam WebSocket.
        """
        if not self.ws:
            return

        print("--- ASR Service: Listening for transcripts... ---")
        try:
            async for message in self.ws:
                if message.type == "speech_start":
                    print("--- ASR Service: Speech detected ---")
                
                elif message.type == "speech_end":
                    print("--- ASR Service: Speech ended ---")
                
                elif message.type == "transcript":
                    if message.text:
                        print(f"--- ASR Service: Transcript received: '{message.text}' ---")
                        await self.on_transcript(message.text)
                        
        except Exception as e:
            print(f"--- ASR Service: Listener error! {e} ---")
            self._is_connected = False

    async def send_audio_chunk(self, audio_chunk: bytes):
        """
        Public method to send a raw audio chunk to Sarvam.
        The 'audio_chunk' is raw L16 PCM data from the client.
        """
        if not self._is_connected or not self.ws:
            print("--- ASR Service: Cannot send, not connected. ---")
            return

        try:
            # Sarvam's SDK expects a base64 encoded string
            encoded_audio = base64.b64encode(audio_chunk).decode("utf-8")
            
            # --- THIS IS THE CORRECTED CALL BASED ON DOCS ---
            # The docs show that 'transcribe' takes the audio data,
            # and the encoding/rate are set on *connect*.
            # However, the Python examples *also* put it in transcribe().
            # Let's follow the specific Python "Basic Streaming" example.
            await self.ws.transcribe(
                audio=encoded_audio,
                # We specify the encoding of the audio *chunk*
                encoding="pcm", # This should match our codec
                sample_rate=self.sample_rate
            )
        except Exception as e:
            print(f"--- ASR Service: Error sending audio! {e} ---")

    async def close(self):
        """
        Shuts down the WebSocket connection and cleans up the listener task.
        """
        if self._listener_task:
            self._listener_task.cancel()
            self._listener_task = None
        if self.ws:
            await self.ws.close()
            self.ws = None
        self._is_connected = False
        print("--- ASR Service: Connection closed. ---")