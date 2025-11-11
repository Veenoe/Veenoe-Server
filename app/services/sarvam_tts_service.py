import asyncio
import base64
from sarvamai import AsyncSarvamAI, AudioOutput
from app.config import settings
from typing import AsyncGenerator

# Initialize the client once, at the module level
client = AsyncSarvamAI(api_subscription_key=settings.SARVAM_API_KEY)

async def text_to_audio_stream(
    text: str, 
    language_code: str = "en-IN", 
    speaker: str = "anushka"
) -> AsyncGenerator[bytes, None]:
    """
    This is a stateless async generator.
    It takes text and yields a stream of audio chunks (bytes).
    """
    print(f"--- TTS Service: Generating audio for: '{text[:30]}...' ---")
    
    # --- FIX: Use 'async with' to correctly handle the context manager ---
    try:
        async with client.text_to_speech_streaming.connect(
            model="bulbul:v2",
            send_completion_event=True # Ask for the "final" event
        ) as ws:

            # 2. Configure the voice
            await ws.configure(
                target_language_code=language_code,
                speaker=speaker,
                output_audio_codec="mp3" # Requesting MP3 output
            )

            # 3. Send the text to be converted
            await ws.convert(text)

            # 4. Tell Sarvam we're done sending text
            await ws.flush()

            # 5. Stream the audio chunks back to the caller
            # This loop will receive audio as it's generated
            async for message in ws:
                if isinstance(message, AudioOutput):
                    # Decode the base64 audio and yield the raw bytes
                    audio_chunk = base64.b64decode(message.data.audio)
                    yield audio_chunk
                
                # The 'final' event tells us the stream is 100% done
                elif message.type == "events" and message.data.event_type == "final":
                    print("--- TTS Service: Received 'final' audio event. ---")
                    break

    except Exception as e:
        print(f"--- TTS Service: Error! {e} ---")
        # In a real app, we might yield a pre-recorded "error" audio chunk
    
    finally:
        # 'async with' handles closing the connection,
        # so we just add a final log.
        print("--- TTS Service: Stream finished and connection closed. ---")