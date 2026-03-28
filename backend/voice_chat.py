# backend/voice_chat.py
"""
REAL Voice Conversation Interface for Kinnect AI.
NO SIMULATION - 100% real audio I/O with LangGraph integration.
"""

import os
import sys
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.audio_handler import AudioHandler
from backend.database import query_memory, add_memory
from backend.graph.prompts import CONVERSATIONAL_SYSTEM_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from config import settings


class KinnectVoiceChat:
    """
    Real voice-based conversation with Kinnect AI.
    
    This is NOT a simulation:
    - Real microphone input → Whisper STT
    - Real Gemini API responses
    - Real gTTS voice output
    - Real ChromaDB memory storage
    """
    
    def __init__(
        self,
        user_id: str,
        whisper_model: str = "base",
        tts_method: str = "gtts"
    ):
        """
        Initialize voice chat.
        
        Args:
            user_id: Patient identifier
            whisper_model: Whisper model size (tiny/base/small)
            tts_method: TTS engine (gtts/pyttsx3)
        """
        self.user_id = user_id
        self.tts_method = tts_method
        
        # Initialize audio handler
        print("\n" + "=" * 70)
        print("KINNECT AI - Voice Conversation System")
        print("=" * 70)
        print(f"Patient ID: {user_id}")
        print("Initializing audio systems...\n")
        
        self.audio = AudioHandler(whisper_model=whisper_model)
        
        # Initialize LLM
        print("🔄 Connecting to Gemini API...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.7
        )
        print("✅ Gemini API connected\n")
        
        # Session data
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.transcript = ""
        self.messages = []
        self.turn_count = 0
        
        # Directories
        os.makedirs("transcripts", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)
    
    def load_patient_context(self) -> tuple[str, list]:
        """
        Load patient memories from ChromaDB (REAL RAG).
        
        Returns:
            (context_summary, memories_list)
        """
        print("=" * 60)
        print("🧠 CONTEXT AGENT - Loading patient memories")
        print("=" * 60)
        
        # Real semantic search in ChromaDB
        query = "recent conversations, important people, daily routine, medical schedule, health concerns"
        memories = query_memory(
            user_id=self.user_id,
            query_text=query,
            n_results=5
        )
        
        print(f"\n📚 Retrieved {len(memories)} relevant memories:")
        for i, mem in enumerate(memories, 1):
            print(f"   {i}. [{mem['metadata']['entity_type']}] {mem['content'][:60]}...")
        
        # Generate context summary using REAL LLM
        if memories:
            memory_text = "\n".join([
                f"- {m['content']} (Type: {m['metadata']['entity_type']})"
                for m in memories
            ])
            
            summary_prompt = f"""Based on these memories about a patient, write a brief 2-3 sentence 
summary of what's important to know for today's check-in call:

{memory_text}

Today is {datetime.now().strftime('%A, %B %d, %Y')}.
Focus on: people to ask about, medications to check, recent events to follow up on.
"""
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            context_summary = response.content
        else:
            context_summary = f"This is your first conversation with {self.user_id}. Focus on building rapport and learning about their daily life, family, and health routines."
        
        print(f"\n📝 Context Summary:")
        print(f"   {context_summary[:150]}...\n")
        
        return context_summary, memories
    
    def generate_agent_response(
        self,
        system_prompt: str,
        user_text: Optional[str] = None
    ) -> str:
        """
        Generate agent response using REAL Gemini API.
        
        Args:
            system_prompt: The system context
            user_text: User's transcribed speech (None for initial greeting)
        
        Returns:
            Agent's response text
        """
        if user_text:
            # Add user message to transcript
            self.transcript += f"User: {user_text}\n"
            self.messages.append({"role": "user", "content": user_text})
        
        # Build conversation context
        if user_text:
            prompt = f"{system_prompt}\n\nConversation so far:\n{self.transcript}\n\nRespond naturally to the user's last message. Keep response concise (2-3 sentences)."
        else:
            prompt = f"{system_prompt}\n\nStart the daily check-in call. Greet the patient warmly and ask how they are doing today."
        
        # REAL API call to Gemini
        response = self.llm.invoke([HumanMessage(content=prompt)])
        agent_text = response.content
        
        # Add to transcript
        self.transcript += f"Agent: {agent_text}\n"
        self.messages.append({"role": "assistant", "content": agent_text})
        
        return agent_text
    
    def speak(self, text: str):
        """
        Convert text to speech and play it (REAL TTS).
        
        Args:
            text: Text to speak
        """
        self.audio.speak_text(text, method=self.tts_method)
    
    def listen(self) -> str:
        """
        Listen for user speech and transcribe (REAL STT).
        
        Returns:
            Transcribed text
        """
        print("\n🎙️ Listening... (speak now, pause when done)")
        
        # Record until silence (REAL microphone input)
        result = self.audio.record_until_silence(
            silence_threshold=0.01,
            silence_duration=1.5,
            max_duration=30.0,
            language="en"
        )
        
        text = result.get("text", "").strip()
        
        if text:
            print(f"👤 You said: \"{text}\"")
        else:
            print("⚠️ Could not understand audio")
        
        return text
    
    def extract_memories(self) -> list:
        """
        Extract new facts from conversation (REAL LLM extraction).
        
        Returns:
            List of extracted entities
        """
        print("\n" + "=" * 60)
        print("📝 MEMORY EXTRACTOR - Analyzing conversation")
        print("=" * 60)
        
        if not self.transcript:
            print("   ⚠️ No transcript to analyze")
            return []
        
        extraction_prompt = f"""Analyze this conversation and extract NEW facts to remember.

TRANSCRIPT:
{self.transcript}

Extract facts in these categories:
1. RELATIONSHIP: People mentioned (family, friends, doctors)
2. SCHEDULE: Regular activities, appointments, medication times
3. HEALTH: Physical or mental health mentions, symptoms
4. PREFERENCE: Likes, dislikes, hobbies
5. EVENT: Specific things that happened or will happen

Return a JSON array:
[
    {{"content": "The exact fact", "entity_type": "relationship|schedule|health|preference|event"}}
]

If no new facts, return: []
Only return the JSON array, nothing else.
"""
        
        response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
        
        try:
            import json
            import re
            
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            entities = json.loads(content)
            
            print(f"\n🆕 Extracted {len(entities)} new facts:")
            
            # Save to ChromaDB (REAL storage)
            for entity in entities:
                if isinstance(entity, dict) and "content" in entity:
                    print(f"   • [{entity.get('entity_type', 'fact')}] {entity['content'][:50]}...")
                    
                    add_memory(
                        user_id=self.user_id,
                        content=entity["content"],
                        entity_type=entity.get("entity_type", "fact"),
                        source="voice_transcript"
                    )
            
            print(f"\n✅ Saved {len(entities)} memories to ChromaDB")
            return entities
            
        except Exception as e:
            print(f"   ⚠️ Extraction error: {str(e)}")
            return []
    
    def run_diagnostic(self, memories: list) -> dict:
        """
        Analyze conversation for cognitive health (REAL diagnostic).
        
        Args:
            memories: Patient's known memories
        
        Returns:
            Diagnostic report dict
        """
        print("\n" + "=" * 60)
        print("🔬 DIAGNOSTIC AGENT - Analyzing cognitive health")
        print("=" * 60)
        
        if not self.transcript:
            return {"cognitive_score": 100, "anomalies": [], "summary": "No conversation to analyze"}
        
        known_facts = "\n".join([
            f"- {m['content']}" for m in memories
        ]) if memories else "No baseline information available."
        
        diagnostic_prompt = f"""Analyze this conversation for cognitive health indicators.

KNOWN FACTS ABOUT PATIENT:
{known_facts}

TODAY'S CONVERSATION:
{self.transcript}

Analyze for:
1. Memory issues (forgetting recent topics, not recognizing names)
2. Communication changes (difficulty finding words, confusion)
3. Behavioral indicators (missed medications, confusion about routines)
4. Positive indicators (clear responses, good recall)

SCORING:
- 90-100: Excellent cognition
- 75-89: Good, minor issues
- 60-74: Fair, some concerning patterns
- 40-59: Concerning, recommend check-in
- Below 40: Urgent attention needed

Return JSON:
{{
    "cognitive_score": <0-100>,
    "anomalies": [
        {{"type": "memory|communication|behavioral", "description": "What was observed", "severity": "low|medium|high"}}
    ],
    "positive_observations": ["list of good signs"],
    "summary": "2-3 sentence assessment"
}}

Only return the JSON, nothing else.
"""
        
        response = self.llm.invoke([HumanMessage(content=diagnostic_prompt)])
        
        try:
            import json
            import re
            
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            report = json.loads(content)
            
            score = report.get("cognitive_score", 75)
            anomalies = report.get("anomalies", [])
            
            print(f"\n📊 Analysis Results:")
            print(f"   Cognitive Score: {score}/100")
            print(f"   Anomalies Found: {len(anomalies)}")
            
            if anomalies:
                print(f"\n   ⚠️ Anomalies Detected:")
                for a in anomalies:
                    if isinstance(a, dict):
                        print(f"      • [{a.get('severity', 'unknown')}] {a.get('description', '')[:50]}...")
            
            if report.get("positive_observations"):
                print(f"\n   ✅ Positive Observations:")
                for obs in report["positive_observations"][:3]:
                    print(f"      • {obs[:50]}...")
            
            print(f"\n   📋 Summary: {report.get('summary', 'N/A')[:100]}...")
            
            # Check if alert needed
            needs_alert = (
                score < 60 or
                any(a.get("severity") == "high" for a in anomalies if isinstance(a, dict))
            )
            
            print(f"\n   🚨 Alert Needed: {needs_alert}")
            report["needs_alert"] = needs_alert
            
            return report
            
        except Exception as e:
            print(f"   ⚠️ Diagnostic error: {str(e)}")
            return {"cognitive_score": 75, "anomalies": [], "summary": "Analysis incomplete", "needs_alert": False}
    
    def send_alert(self, diagnostic_report: dict):
        """
        Send alert to caregiver if needed (REAL alert generation).
        
        Args:
            diagnostic_report: The diagnostic analysis results
        """
        print("\n" + "=" * 60)
        print("🚨 ALERT AGENT - Generating caregiver notification")
        print("=" * 60)
        
        alert_prompt = f"""Generate a professional but warm alert message for a caregiver.

PATIENT: {self.user_id}
DATE: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
COGNITIVE SCORE: {diagnostic_report.get('cognitive_score', 'N/A')}/100

DIAGNOSTIC REPORT:
{diagnostic_report}

CONVERSATION EXCERPT:
{self.transcript[-500:] if len(self.transcript) > 500 else self.transcript}

Write an email-style message that:
1. Opens with clear but non-alarming subject line
2. Summarizes what was observed (be specific but compassionate)
3. Suggests possible next steps
4. Ends with reassurance

Keep it professional but human.
"""
        
        response = self.llm.invoke([HumanMessage(content=alert_prompt)])
        alert_message = response.content
        
        print(f"\n📧 Generated Alert Message:")
        print("-" * 40)
        print(alert_message[:500])
        if len(alert_message) > 500:
            print("...")
        print("-" * 40)
        
        print(f"\n✅ Alert would be sent to caregiver for {self.user_id}")
        
        # Save alert to file
        alert_file = f"transcripts/alert_{self.user_id}_{self.session_id}.txt"
        with open(alert_file, 'w', encoding='utf-8') as f:
            f.write(alert_message)
        print(f"💾 Alert saved to: {alert_file}")
    
    def save_transcript(self, diagnostic_report: dict, new_entities: list):
        """
        Save complete session transcript.
        
        Args:
            diagnostic_report: Diagnostic results
            new_entities: Extracted memories
        """
        filename = f"transcripts/voice_{self.user_id}_{self.session_id}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("KINNECT AI - Voice Conversation Transcript\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Patient ID: {self.user_id}\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Conversation Turns: {self.turn_count}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("CONVERSATION:\n")
            f.write("-" * 70 + "\n")
            f.write(self.transcript)
            f.write("\n" + "-" * 70 + "\n\n")
            
            f.write("DIAGNOSTIC SUMMARY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Cognitive Score: {diagnostic_report.get('cognitive_score', 'N/A')}/100\n")
            f.write(f"Alert Triggered: {diagnostic_report.get('needs_alert', False)}\n")
            f.write(f"Summary: {diagnostic_report.get('summary', 'N/A')}\n\n")
            
            if new_entities:
                f.write("NEW MEMORIES EXTRACTED:\n")
                f.write("-" * 70 + "\n")
                for entity in new_entities:
                    f.write(f"  [{entity.get('entity_type', 'unknown')}] {entity.get('content', 'N/A')}\n")
        
        print(f"\n💾 Transcript saved to: {filename}")
    
    def run_conversation(self):
        """
        Run the complete voice conversation.
        
        This is the REAL conversation loop:
        1. Load context (RAG)
        2. Agent greets (TTS)
        3. User speaks (STT)
        4. Agent responds (LLM + TTS)
        5. Repeat until user says goodbye
        6. Extract memories (LLM)
        7. Run diagnostics (LLM)
        8. Send alert if needed
        """
        print("\n" + "=" * 70)
        print("🎙️ STARTING VOICE CONVERSATION")
        print("=" * 70)
        print("Say 'goodbye', 'bye', or 'end' to finish the call.\n")
        
        # =====================================================================
        # PHASE 1: Load Context (REAL RAG)
        # =====================================================================
        context_summary, memories = self.load_patient_context()
        
        # Build system prompt
        memories_formatted = "\n".join([
            f"- {m['content']}" for m in memories
        ]) if memories else "No prior memories available."
        
        system_prompt = CONVERSATIONAL_SYSTEM_PROMPT.format(
            user_id=self.user_id,
            context_summary=context_summary,
            memories_formatted=memories_formatted,
            current_date=datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
        )
        
        # =====================================================================
        # PHASE 2: Agent Greeting (REAL LLM + TTS)
        # =====================================================================
        print("\n📞 Calling patient...\n")
        
        agent_greeting = self.generate_agent_response(system_prompt, user_text=None)
        print(f"🤖 Agent: {agent_greeting}\n")
        self.speak(agent_greeting)
        
        # =====================================================================
        # PHASE 3: Conversation Loop (REAL STT + LLM + TTS)
        # =====================================================================
        end_phrases = ['bye', 'goodbye', 'end', 'quit', 'exit', 'hang up', 'end call']
        
        while True:
            # REAL microphone input + transcription
            user_text = self.listen()
            
            if not user_text:
                self.speak("I'm sorry, I didn't catch that. Could you please repeat?")
                continue
            
            self.turn_count += 1
            
            # Check for end of conversation
            if any(phrase in user_text.lower() for phrase in end_phrases):
                # Generate farewell
                farewell = self.generate_agent_response(
                    system_prompt,
                    "I need to go now, goodbye."
                )
                print(f"\n🤖 Agent: {farewell}\n")
                self.speak(farewell)
                break
            
            # REAL LLM response
            agent_response = self.generate_agent_response(system_prompt, user_text)
            print(f"\n🤖 Agent: {agent_response}\n")
            
            # REAL TTS output
            self.speak(agent_response)
        
        # =====================================================================
        # PHASE 4: Post-Call Processing
        # =====================================================================
        print("\n📞 Call ended.\n")
        print("=" * 70)
        print("POST-CALL PROCESSING")
        print("=" * 70)
        
        # Memory Extraction (REAL LLM)
        new_entities = self.extract_memories()
        
        # Diagnostic Analysis (REAL LLM)
        diagnostic_report = self.run_diagnostic(memories)
        
        # Alert if needed
        if diagnostic_report.get("needs_alert", False):
            self.send_alert(diagnostic_report)
        
        # Save transcript
        self.save_transcript(diagnostic_report, new_entities)
        
        # =====================================================================
        # PHASE 5: Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("CONVERSATION SUMMARY")
        print("=" * 70)
        print(f"Duration: {self.turn_count} conversation turns")
        print(f"Transcript length: {len(self.transcript)} characters")
        print(f"New memories saved: {len(new_entities)}")
        print(f"Cognitive score: {diagnostic_report.get('cognitive_score', 'N/A')}/100")
        print(f"Alert triggered: {diagnostic_report.get('needs_alert', False)}")
        print("=" * 70 + "\n")


def main():
    """Main entry point for voice chat."""
    import sys
    
    # Get user ID
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = input("Enter patient ID (or press Enter for 'voice_patient'): ").strip()
        if not user_id:
            user_id = "voice_patient"
    
    # Optional: whisper model selection
    whisper_model = "base"  # Options: tiny, base, small, medium, large
    
    # Optional: TTS method selection
    tts_method = "gtts"  # Options: gtts (natural, online), pyttsx3 (robotic, offline)
    
    # Create and run voice chat
    chat = KinnectVoiceChat(
        user_id=user_id,
        whisper_model=whisper_model,
        tts_method=tts_method
    )
    
    try:
        chat.run_conversation()
    except KeyboardInterrupt:
        print("\n\n⚠️ Conversation interrupted by user")
        print("Saving partial transcript...\n")
        chat.save_transcript(
            {"cognitive_score": "N/A", "needs_alert": False, "summary": "Interrupted"},
            []
        )


if __name__ == "__main__":
    main()