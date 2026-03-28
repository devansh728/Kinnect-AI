# backend/audio_handler.py
"""
Audio processing module for Kinnect AI.
Handles Speech-to-Text (STT) and Text-to-Speech (TTS).
"""

import whisper
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from datetime import datetime
import tempfile
import os
from gtts import gTTS
import pyttsx3
import pygame
from io import BytesIO


class AudioHandler:
    """Manages audio input/output for Kinnect AI."""

    def __init__(self, whisper_model: str = "base"):
        """
        Initialize audio handler.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
                          - tiny: Fastest, least accurate (~1GB RAM)
                          - base: Good balance (~1GB RAM) ← RECOMMENDED
                          - small: Better accuracy (~2GB RAM)
                          - medium: High accuracy (~5GB RAM)
                          - large: Best accuracy (~10GB RAM)
        """
        print(f"🔄 Loading Whisper model: {whisper_model}...")
        self.whisper_model = whisper.load_model(whisper_model)
        print(f"✅ Whisper model loaded\n")

        # Audio settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1  # Mono audio
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self._configure_tts_engine()

        # Recording settings
        self.is_recording = False
        self.audio_buffer = []

        # Initialize pygame for audio playback
        pygame.mixer.init()

    def _configure_tts_engine(self):
        """Configure pyttsx3 TTS engine for elderly-friendly speech."""
        # Set voice properties
        self.tts_engine.setProperty('rate', 150)  # Slower speed for clarity (default: 200)
        self.tts_engine.setProperty('volume', 1.0)  # Maximum volume

        # Try to select a clear, friendly voice
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Prefer female voices (often clearer for elderly users)
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break

    # =========================================================================
    # SPEECH-TO-TEXT METHODS
    # =========================================================================

    def transcribe_file(self, audio_file_path: str, language: str = "en") -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_file_path: Path to audio file (wav, mp3, m4a, etc.)
            language: Language code (en, es, fr, etc.) or None for auto-detect

        Returns:
            {
                "text": "transcribed text",
                "language": "detected language",
                "confidence": 0.95,
                "duration": 5.2
            }
        """
        print(f"🎤 Transcribing: {audio_file_path}")

        try:
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_file_path,
                language=language,
                fp16=False  # Disable FP16 for CPU compatibility
            )

            transcription = {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "confidence": self._calculate_confidence(result),
                "duration": self._get_audio_duration(audio_file_path)
            }

            print(f"✅ Transcription: \"{transcription['text'][:100]}...\"")
            print(f"   Language: {transcription['language']}")
            print(f"   Confidence: {transcription['confidence']:.2%}")
            print(f"   Duration: {transcription['duration']:.1f}s\n")

            return transcription

        except Exception as e:
            print(f"❌ Transcription error: {str(e)}\n")
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "duration": 0.0,
                "error": str(e)
            }

    def transcribe_numpy(self, audio_array: np.ndarray, language: str = "en") -> dict:
        """
        Transcribe a numpy audio array to text.

        Args:
            audio_array: Numpy array of audio samples
            language: Language code

        Returns:
            Same format as transcribe_file()
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_array, self.sample_rate)

        try:
            result = self.transcribe_file(tmp_path, language)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def record_and_transcribe(
        self,
        duration: float = 5.0,
        language: str = "en"
    ) -> dict:
        """
        Record audio from microphone and transcribe.

        Args:
            duration: Recording duration in seconds
            language: Language code

        Returns:
            Transcription dict
        """
        print(f"🎙️ Recording for {duration} seconds...")
        print("   (Speak now...)\n")

        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to finish

        print("✅ Recording complete\n")

        # Transcribe
        return self.transcribe_numpy(audio_data.flatten(), language)

    def record_until_silence(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
        language: str = "en"
    ) -> dict:
        """
        Record audio until silence is detected (for natural conversation).

        Args:
            silence_threshold: Volume threshold to detect silence (0.0-1.0)
            silence_duration: How long silence before stopping (seconds)
            max_duration: Maximum recording length (seconds)
            language: Language code

        Returns:
            Transcription dict
        """
        print("🎙️ Recording (will stop when you stop speaking)...")
        print("   (Speak now...)\n")

        audio_buffer = []
        silence_counter = 0
        silence_chunks = int(silence_duration * self.sample_rate / 1024)
        max_chunks = int(max_duration * self.sample_rate / 1024)
        chunk_count = 0

        def audio_callback(indata, frames, time, status):
            nonlocal silence_counter, chunk_count

            # Calculate volume
            volume = np.abs(indata).mean()

            # Store audio
            audio_buffer.append(indata.copy())
            chunk_count += 1

            # Check for silence
            if volume < silence_threshold:
                silence_counter += 1
            else:
                silence_counter = 0

            # Stop if silence detected or max duration reached
            if silence_counter >= silence_chunks or chunk_count >= max_chunks:
                raise sd.CallbackStop()

        # Start recording
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=audio_callback,
            blocksize=1024
        ):
            try:
                sd.sleep(int(max_duration * 1000))
            except sd.CallbackStop:
                pass

        print("✅ Recording complete\n")

        # Convert to numpy array
        if audio_buffer:
            audio_array = np.concatenate(audio_buffer, axis=0).flatten()
            return self.transcribe_numpy(audio_array, language)
        else:
            return {
                "text": "",
                "language": language,
                "confidence": 0.0,
                "duration": 0.0,
                "error": "No audio recorded"
            }

    # =========================================================================
    # TEXT-TO-SPEECH METHODS
    # =========================================================================

    def text_to_speech_gtts(
        self,
        text: str,
        language: str = "en",
        slow: bool = False,
        save_path: str = None
    ) -> str:
        """
        Convert text to speech using gTTS (Google Text-to-Speech).
        Requires internet connection but produces natural-sounding speech.

        Args:
            text: Text to convert to speech
            language: Language code (en, es, fr, etc.)
            slow: If True, speaks more slowly
            save_path: Path to save audio file (optional)

        Returns:
            Path to generated audio file
        """
        print(f"🔊 Generating speech (gTTS): \"{text[:50]}...\"")

        try:
            # Generate speech
            tts = gTTS(text=text, lang=language, slow=slow)

            # Save to file
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"tts_output/gtts_{timestamp}.mp3"

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tts.save(save_path)

            print(f"✅ Speech generated: {save_path}\n")
            return save_path

        except Exception as e:
            print(f"❌ gTTS error: {str(e)}\n")
            return None

    def text_to_speech_pyttsx3(
        self,
        text: str,
        save_path: str = None
    ) -> str:
        """
        Convert text to speech using pyttsx3 (offline TTS).
        Works offline but less natural than gTTS.

        Args:
            text: Text to convert to speech
            save_path: Path to save audio file (optional)

        Returns:
            Path to generated audio file
        """
        print(f"🔊 Generating speech (pyttsx3): \"{text[:50]}...\"")

        try:
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"tts_output/pyttsx3_{timestamp}.wav"

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Generate and save
            self.tts_engine.save_to_file(text, save_path)
            self.tts_engine.runAndWait()

            print(f"✅ Speech generated: {save_path}\n")
            return save_path

        except Exception as e:
            print(f"❌ pyttsx3 error: {str(e)}\n")
            return None

    def speak_text(self, text: str, method: str = "gtts"):
        """
        Generate speech and play it immediately.

        Args:
            text: Text to speak
            method: "gtts" (natural, online) or "pyttsx3" (robotic, offline)
        """
        print(f"🔊 Speaking: \"{text[:50]}...\"")

        # Generate audio file
        if method == "gtts":
            audio_file = self.text_to_speech_gtts(text)
        else:
            audio_file = self.text_to_speech_pyttsx3(text)

        if audio_file:
            # Play audio
            self.play_audio_file(audio_file)

            # Clean up temp file
            try:
                os.remove(audio_file)
            except:
                pass

    def play_audio_file(self, audio_file_path: str):
        """
        Play an audio file through speakers.

        Args:
            audio_file_path: Path to audio file
        """
        print(f"▶️ Playing audio: {audio_file_path}")

        try:
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            print("✅ Playback complete\n")

        except Exception as e:
            print(f"❌ Playback error: {str(e)}\n")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_confidence(self, whisper_result: dict) -> float:
        """
        Calculate confidence score from Whisper result.
        Whisper doesn't provide confidence directly, so we estimate it.
        """
        # Use average log probability as confidence proxy
        if "segments" in whisper_result:
            segments = whisper_result["segments"]
            if segments:
                avg_logprob = sum(s.get("avg_logprob", -1.0)
                                  for s in segments) / len(segments)
                # Convert log probability to 0-1 scale (rough approximation)
                confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))
                return confidence

        return 0.8  # Default confidence if we can't calculate

    def _get_audio_duration(self, audio_file_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            info = sf.info(audio_file_path)
            return info.duration
        except:
            return 0.0

    def save_recording(self, audio_array: np.ndarray, filename: str = None) -> str:
        """
        Save audio array to file.

        Args:
            audio_array: Numpy audio array
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/recording_{timestamp}.wav"

        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save audio
        sf.write(filename, audio_array, self.sample_rate)
        print(f"💾 Audio saved to: {filename}\n")

        return filename