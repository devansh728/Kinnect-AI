# backend/websocket/audio_streamer.py
import base64
import os
import tempfile
from backend.audio_handler import AudioHandler

def bytes_to_base64(audio_bytes: bytes) -> str:
    """Convert raw bytes to base64 encoded string."""
    return base64.b64encode(audio_bytes).decode("utf-8")

def base64_to_bytes(b64_str: str) -> bytes:
    """Convert base64 encoded string to raw bytes."""
    return base64.b64decode(b64_str)

def transcribe_from_base64(b64_audio: str, audio_handler: AudioHandler) -> str:
    """
    Decodes base64 audio data, writes it to a temporary file,
    and transcribes it to text using Whisper.
    """
    audio_bytes = base64_to_bytes(b64_audio)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(audio_bytes)
        
    try:
        result = audio_handler.transcribe_file(tmp_path)
        return result.get("text", "")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def synthesize_to_base64(text: str, audio_handler: AudioHandler, method: str = "gtts") -> str:
    """
    Converts text to speech using the selected method (gtts/pyttsx3),
    reads the generated audio file, encodes it as base64, and cleans up the temp file.
    """
    audio_file = None
    if method == "gtts":
        audio_file = audio_handler.text_to_speech_gtts(text)
    else:
        audio_file = audio_handler.text_to_speech_pyttsx3(text)
        
    if not audio_file or not os.path.exists(audio_file):
        return ""
        
    try:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        return bytes_to_base64(audio_bytes)
    finally:
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception:
                pass
