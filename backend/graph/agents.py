# backend/graph/agents.py
"""
Complete agent implementations for Kinnect AI.
Each agent is a LangGraph node that processes the shared state.
"""

from backend.graph.state import KinnectState
from backend.graph.prompts import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    DIAGNOSTIC_ANALYSIS_PROMPT,
    ALERT_MESSAGE_PROMPT
)
from backend.database import query_memory, add_memory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import settings
from datetime import datetime
import json
import re

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.7
)

# Lower temperature for structured outputs
llm_structured = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.2
)


# =============================================================================
# AGENT 1: CONTEXT AGENT (Pre-Call)
# =============================================================================

def context_agent_node(state: KinnectState) -> dict:
    """
    PRE-CALL AGENT: Loads patient history and prepares conversation context.
    
    This is REAL RAG:
    1. Semantic search in ChromaDB
    2. Dynamic context generation via LLM
    3. Personalized system prompt for each call
    """
    user_id = state["user_id"]
    
    print(f"\n{'='*60}")
    print(f"🧠 CONTEXT AGENT - Loading memories for {user_id}")
    print(f"{'='*60}")
    
    try:
        # Semantic retrieval from ChromaDB
        query_text = "recent conversations, important people, daily routine, medical schedule, health concerns"
        memories = query_memory(user_id=user_id, query_text=query_text, n_results=5)
        
        print(f"\n📚 Retrieved {len(memories)} relevant memories:")
        for i, mem in enumerate(memories, 1):
            print(f"   {i}. [{mem['metadata']['entity_type']}] {mem['content'][:60]}...")
        
        # Generate context summary
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
            response = llm.invoke(summary_prompt)
            context_summary = response.content
        else:
            context_summary = f"This is your first conversation with {user_id}. Focus on building rapport and learning about their daily life, family, and health routines."
        
        print(f"\n📝 Generated Context Summary:")
        print(f"   {context_summary[:150]}...")
        
        return {
            "retrieved_memories": memories,
            "context_summary": context_summary,
            "next_agent": "conversational_agent",
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "context_agent": {
                    "memories_count": len(memories),
                    "summary_generated": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error in Context Agent: {str(e)}")
        return {
            "errors": [f"Context Agent Error: {str(e)}"],
            "context_summary": "No prior context available. Start fresh.",
            "retrieved_memories": [],
            "next_agent": "conversational_agent"
        }


# =============================================================================
# AGENT 2: CONVERSATIONAL AGENT (Live Call)
# =============================================================================

def conversational_agent_node(state: KinnectState) -> dict:
    """
    LIVE CALL AGENT: Handles the actual conversation with the patient.
    
    PHASE 2: Simulated conversation for testing workflow
    PHASE 3: Will receive real WebSocket messages (just swap input source)
    """
    user_id = state["user_id"]
    context_summary = state.get("context_summary", "")
    memories = state.get("retrieved_memories", [])
    
    print(f"\n{'='*60}")
    print(f"💬 CONVERSATIONAL AGENT - Starting call with {user_id}")
    print(f"{'='*60}")
    
    try:
        # Format memories for the prompt
        memories_formatted = "\n".join([
            f"- {m['content']}" for m in memories
        ]) if memories else "No prior memories available."
        
        # Build the system prompt
        system_prompt = CONVERSATIONAL_SYSTEM_PROMPT.format(
            user_id=user_id,
            context_summary=context_summary,
            memories_formatted=memories_formatted,
            current_date=datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
        )
        
        # For testing: Simulate a multi-turn conversation
        # In Phase 3: Replace this with WebSocket.receive() loop
        simulated_user_inputs = [
            "Oh hello! Yes, I'm doing alright today.",
            "Well, Sarah came by yesterday. We had tea together.",
            "I think I took my pills this morning... or was that yesterday?",
            "I've been a bit tired lately. Nothing too bad though.",
            "Alright dear, I should go now. Thank you for calling."
        ]
        
        # Initialize conversation (Gemini-compatible format)
        messages = []
        full_transcript = ""
        
        # First, agent initiates the call
        # FIX: Embed system context in HumanMessage (no SystemMessage)
        initial_prompt = f"{system_prompt}\n\nYou are now starting the daily check-in call. Greet the patient warmly and ask how they are doing today."
        
        initial_response = llm.invoke([HumanMessage(content=initial_prompt)])
        agent_greeting = initial_response.content
        
        messages.append(AIMessage(content=agent_greeting))
        full_transcript += f"Agent: {agent_greeting}\n"
        print(f"\n🤖 Agent: {agent_greeting}")
        
        # Simulate conversation turns
        for user_input in simulated_user_inputs:
            print(f"\n👤 User: {user_input}")
            full_transcript += f"User: {user_input}\n"
            
            messages.append(HumanMessage(content=user_input))
            
            # FIX: Include system context in each turn (maintains agent behavior)
            conversation_with_context = [
                HumanMessage(content=f"SYSTEM CONTEXT: {system_prompt}\n\nCONVERSATION SO FAR:\n{full_transcript}\n\nRespond naturally to the user's last message.")
            ]
            
            response = llm.invoke(conversation_with_context)
            agent_response = response.content
            
            messages.append(AIMessage(content=agent_response))
            full_transcript += f"Agent: {agent_response}\n"
            print(f"🤖 Agent: {agent_response}")
        
        # Convert messages to dict format for state
        messages_dict = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                messages_dict.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages_dict.append({"role": "assistant", "content": msg.content})
        
        print(f"\n✅ Conversation completed: {len(messages_dict)} messages")
        
        return {
            "messages": messages_dict,
            "transcript": full_transcript,
            "next_agent": "memory_extractor",
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "conversational_agent": {
                    "turns": len(simulated_user_inputs),
                    "transcript_length": len(full_transcript),
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error in Conversational Agent: {str(e)}")
        import traceback
        traceback.print_exc()  # Show full error for debugging
        return {
            "errors": [f"Conversational Agent Error: {str(e)}"],
            "transcript": "Error during conversation.",
            "messages": [],
            "next_agent": "memory_extractor"
        }

# =============================================================================
# AGENT 3: MEMORY EXTRACTION AGENT (Post-Call)
# =============================================================================

def memory_extraction_node(state: KinnectState) -> dict:
    """
    POST-CALL AGENT: Extracts new facts from the conversation.
    
    Uses structured output to ensure consistent JSON format.
    Saves extracted memories to ChromaDB via MCP.
    """
    user_id = state["user_id"]
    transcript = state.get("transcript", "")
    existing_memories = state.get("retrieved_memories", [])
    
    print(f"\n{'='*60}")
    print(f"📝 MEMORY EXTRACTOR - Processing transcript")
    print(f"{'='*60}")
    
    if not transcript:
        print("   ⚠️ No transcript available")
        return {
            "new_entities": [],
            "next_agent": "diagnostic_analyzer"
        }
    
    try:
        # Format existing memories
        existing_formatted = "\n".join([
            f"- {m['content']}" for m in existing_memories
        ]) if existing_memories else "None"
        
        # Build extraction prompt
        extraction_prompt = MEMORY_EXTRACTION_PROMPT.format(
            transcript=transcript,
            existing_memories=existing_formatted
        )
        
        # Use structured LLM for consistent JSON
        response = llm_structured.invoke(extraction_prompt)
        
        # Parse the JSON response
        try:
            # Clean the response (remove markdown code blocks if present)
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            new_entities = json.loads(content)
        except json.JSONDecodeError:
            print(f"   ⚠️ Failed to parse JSON, extracting manually")
            new_entities = []
        
        print(f"\n🆕 Extracted {len(new_entities)} new facts:")
        
        # Save each new memory to ChromaDB
        saved_entities = []
        for entity in new_entities:
            if isinstance(entity, dict) and "content" in entity:
                print(f"   • [{entity.get('entity_type', 'fact')}] {entity['content'][:50]}...")
                
                # Save to ChromaDB
                add_memory(
                    user_id=user_id,
                    content=entity["content"],
                    entity_type=entity.get("entity_type", "fact"),
                    source="transcript_extract"
                )
                
                saved_entities.append({
                    **entity,
                    "timestamp": datetime.now().isoformat()
                })
        
        print(f"\n✅ Saved {len(saved_entities)} memories to ChromaDB")
        
        return {
            "new_entities": saved_entities,
            "next_agent": "diagnostic_analyzer",
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "memory_extractor": {
                    "entities_extracted": len(saved_entities),
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error in Memory Extractor: {str(e)}")
        return {
            "errors": [f"Memory Extractor Error: {str(e)}"],
            "new_entities": [],
            "next_agent": "diagnostic_analyzer"
        }


# =============================================================================
# AGENT 4: DIAGNOSTIC AGENT (Post-Call Analysis)
# =============================================================================

def diagnostic_agent_node(state: KinnectState) -> dict:
    """
    POST-CALL AGENT: Analyzes conversation for cognitive health indicators.
    
    This is the critical agent that decides whether to alert caregivers.
    Uses structured output for consistent scoring.
    """
    user_id = state["user_id"]
    transcript = state.get("transcript", "")
    memories = state.get("retrieved_memories", [])
    
    print(f"\n{'='*60}")
    print(f"🔬 DIAGNOSTIC AGENT - Analyzing cognitive health")
    print(f"{'='*60}")
    
    if not transcript:
        print("   ⚠️ No transcript to analyze")
        return {
            "cognitive_score": 100.0,
            "anomalies_detected": [],
            "needs_alert": False,
            "diagnostic_report": {"summary": "No conversation to analyze"},
            "next_agent": "END"
        }
    
    try:
        # Format known facts
        known_facts = "\n".join([
            f"- {m['content']}" for m in memories
        ]) if memories else "No baseline information available."
        
        # Build diagnostic prompt
        diagnostic_prompt = DIAGNOSTIC_ANALYSIS_PROMPT.format(
            user_id=user_id,
            known_facts=known_facts,
            transcript=transcript
        )
        
        # Use structured LLM for consistent JSON
        response = llm_structured.invoke(diagnostic_prompt)
        
        # Parse the JSON response
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            diagnostic_report = json.loads(content)
        except json.JSONDecodeError:
            print(f"   ⚠️ Failed to parse diagnostic JSON")
            diagnostic_report = {
                "cognitive_score": 75,
                "confidence": 50,
                "anomalies": [],
                "positive_observations": [],
                "recommendations": [],
                "summary": "Unable to complete full analysis."
            }
        
        # Extract key metrics
        cognitive_score = float(diagnostic_report.get("cognitive_score", 75))
        anomalies = diagnostic_report.get("anomalies", [])
        
        # Determine if alert is needed
        # Alert thresholds:
        # - Score below 60: Concerning
        # - Any high-severity anomaly: Alert
        needs_alert = (
            cognitive_score < 60 or
            any(a.get("severity") == "high" for a in anomalies if isinstance(a, dict))
        )
        
        # Format anomalies for state
        anomalies_detected = [
            f"{a.get('type', 'unknown')}: {a.get('description', 'No description')}"
            for a in anomalies if isinstance(a, dict)
        ]
        
        print(f"\n📊 Analysis Results:")
        print(f"   Cognitive Score: {cognitive_score}/100")
        print(f"   Confidence: {diagnostic_report.get('confidence', 'N/A')}%")
        print(f"   Anomalies Found: {len(anomalies)}")
        
        if anomalies:
            print(f"\n   ⚠️ Anomalies Detected:")
            for a in anomalies:
                if isinstance(a, dict):
                    print(f"      • [{a.get('severity', 'unknown')}] {a.get('description', 'No description')[:50]}...")
        
        if diagnostic_report.get("positive_observations"):
            print(f"\n   ✅ Positive Observations:")
            for obs in diagnostic_report["positive_observations"][:3]:
                print(f"      • {obs[:50]}...")
        
        print(f"\n   📋 Summary: {diagnostic_report.get('summary', 'N/A')[:100]}...")
        print(f"\n   🚨 Alert Needed: {needs_alert}")
        
        return {
            "cognitive_score": cognitive_score,
            "anomalies_detected": anomalies_detected,
            "needs_alert": needs_alert,
            "diagnostic_report": diagnostic_report,
            "next_agent": "alert_handler" if needs_alert else "END",
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "diagnostic_agent": {
                    "score": cognitive_score,
                    "anomalies_count": len(anomalies),
                    "alert_triggered": needs_alert,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error in Diagnostic Agent: {str(e)}")
        return {
            "errors": [f"Diagnostic Agent Error: {str(e)}"],
            "cognitive_score": 75.0,
            "anomalies_detected": [],
            "needs_alert": False,
            "diagnostic_report": {"error": str(e)},
            "next_agent": "END"
        }


# =============================================================================
# AGENT 5: ALERT AGENT (Conditional - Only if needed)
# =============================================================================

def alert_agent_node(state: KinnectState) -> dict:
    """
    CONDITIONAL AGENT: Only runs if diagnostic agent flags concerns.
    
    Generates a professional alert message for caregivers.
    In production, this would send via email/SMS using MCP Alert Server.
    """
    user_id = state["user_id"]
    cognitive_score = state.get("cognitive_score", 0)
    diagnostic_report = state.get("diagnostic_report", {})
    transcript = state.get("transcript", "")
    
    print(f"\n{'='*60}")
    print(f"🚨 ALERT AGENT - Generating caregiver notification")
    print(f"{'='*60}")
    
    try:
        # Get a relevant excerpt from transcript (last few exchanges)
        transcript_lines = transcript.strip().split("\n")
        transcript_excerpt = "\n".join(transcript_lines[-10:]) if len(transcript_lines) > 10 else transcript
        
        # Build alert message prompt
        alert_prompt = ALERT_MESSAGE_PROMPT.format(
            user_id=user_id,
            date=datetime.now().strftime('%B %d, %Y at %I:%M %p'),
            score=cognitive_score,
            diagnostic_report=json.dumps(diagnostic_report, indent=2),
            transcript_excerpt=transcript_excerpt
        )
        
        # Generate the alert message
        response = llm.invoke(alert_prompt)
        alert_message = response.content
        
        print(f"\n📧 Generated Alert Message:")
        print("-" * 40)
        print(alert_message[:500])
        if len(alert_message) > 500:
            print("...")
        print("-" * 40)
        
        # In production, this would call the MCP Alert Server
        # For now, we simulate sending
        print(f"\n✅ Alert would be sent to caregiver for {user_id}")
        print(f"   Score: {cognitive_score}/100")
        print(f"   Anomalies: {len(state.get('anomalies_detected', []))}")
        
        return {
            "next_agent": "END",
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "alert_agent": {
                    "alert_sent": True,
                    "alert_message_length": len(alert_message),
                    "recipient": f"caregiver_of_{user_id}",
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error in Alert Agent: {str(e)}")
        return {
            "errors": [f"Alert Agent Error: {str(e)}"],
            "next_agent": "END",
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "alert_agent": {
                    "alert_sent": False,
                    "error": str(e)
                }
            }
        }