# backend/graph/agents_voice.py
"""
Voice-enabled agent implementations for Kinnect AI.
These agents work with REAL audio I/O - no simulation.
"""

from backend.graph.state import KinnectState
from backend.graph.prompts import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    DIAGNOSTIC_ANALYSIS_PROMPT,
    ALERT_MESSAGE_PROMPT
)
from backend.database import query_memory, add_memory
from backend.audio_handler import AudioHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import settings
from datetime import datetime
import json
import re

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.7
)

llm_structured = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.2
)


def voice_conversational_agent(
    state: KinnectState,
    audio_handler: AudioHandler,
    tts_method: str = "gtts"
) -> dict:
    """
    REAL VOICE CONVERSATIONAL AGENT.
    
    NO SIMULATION:
    - Real microphone input via AudioHandler
    - Real Whisper transcription
    - Real Gemini responses
    - Real gTTS speech output
    """
    user_id = state["user_id"]
    context_summary = state.get("context_summary", "")
    memories = state.get("retrieved_memories", [])
    
    print(f"\n{'='*60}")
    print(f"💬 VOICE CONVERSATIONAL AGENT - Call with {user_id}")
    print(f"{'='*60}")
    
    # Build system prompt
    memories_formatted = "\n".join([
        f"- {m['content']}" for m in memories
    ]) if memories else "No prior memories available."
    
    system_prompt = CONVERSATIONAL_SYSTEM_PROMPT.format(
        user_id=user_id,
        context_summary=context_summary,
        memories_formatted=memories_formatted,
        current_date=datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
    )
    
    messages = []
    full_transcript = ""
    turn_count = 0
    
    # Agent greeting
    greeting_prompt = f"{system_prompt}\n\nStart the daily check-in call. Greet the patient warmly."
    response = llm.invoke([HumanMessage(content=greeting_prompt)])
    agent_greeting = response.content
    
    messages.append({"role": "assistant", "content": agent_greeting})
    full_transcript += f"Agent: {agent_greeting}\n"
    
    print(f"\n🤖 Agent: {agent_greeting}")
    audio_handler.speak_text(agent_greeting, method=tts_method)
    
    # Conversation loop with REAL audio
    end_phrases = ['bye', 'goodbye', 'end', 'quit', 'exit', 'hang up']
    
    while turn_count < 20:  # Max turns safety limit
        # REAL microphone input
        print("\n🎙️ Listening...")
        result = audio_handler.record_until_silence(
            silence_threshold=0.01,
            silence_duration=1.5,
            max_duration=30.0
        )
        
        user_text = result.get("text", "").strip()
        
        if not user_text:
            repeat_msg = "I'm sorry, I didn't catch that. Could you please repeat?"
            audio_handler.speak_text(repeat_msg, method=tts_method)
            continue
        
        print(f"👤 User: {user_text}")
        messages.append({"role": "user", "content": user_text})
        full_transcript += f"User: {user_text}\n"
        turn_count += 1
        
        # Check for end
        if any(phrase in user_text.lower() for phrase in end_phrases):
            farewell_prompt = f"{system_prompt}\n\nConversation:\n{full_transcript}\n\nThe user is ending the call. Say a warm goodbye."
            response = llm.invoke([HumanMessage(content=farewell_prompt)])
            farewell = response.content
            
            messages.append({"role": "assistant", "content": farewell})
            full_transcript += f"Agent: {farewell}\n"
            
            print(f"\n🤖 Agent: {farewell}")
            audio_handler.speak_text(farewell, method=tts_method)
            break
        
        # REAL LLM response
        response_prompt = f"{system_prompt}\n\nConversation:\n{full_transcript}\n\nRespond naturally. Keep it concise (2-3 sentences)."
        response = llm.invoke([HumanMessage(content=response_prompt)])
        agent_response = response.content
        
        messages.append({"role": "assistant", "content": agent_response})
        full_transcript += f"Agent: {agent_response}\n"
        
        print(f"\n🤖 Agent: {agent_response}")
        audio_handler.speak_text(agent_response, method=tts_method)
    
    print(f"\n✅ Voice conversation completed: {len(messages)} messages")
    
    return {
        "messages": messages,
        "transcript": full_transcript,
        "next_agent": "memory_extractor",
        "agent_outputs": {
            **state.get("agent_outputs", {}),
            "conversational_agent": {
                "turns": turn_count,
                "transcript_length": len(full_transcript),
                "mode": "voice",
                "timestamp": datetime.now().isoformat()
            }
        }
    }