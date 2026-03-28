# backend/graph/state_utils.py
from backend.graph.state import KinnectState
from datetime import datetime
import uuid

def create_initial_state(user_id: str) -> KinnectState:
    """
    Factory function to create a fresh state object for a new call.
    This ensures all required fields are initialized properly.
    """
    return {
        "user_id": user_id,
        "session_id": str(uuid.uuid4()),
        "timestamp": datetime.now(),
        "messages": [],
        "transcript": "",
        "audio_chunks": [],
        "retrieved_memories": [],
        "new_entities": [],
        "context_summary": "",
        "cognitive_score": 100.0,  # Start optimistic
        "anomalies_detected": [],
        "needs_alert": False,
        "diagnostic_report": {},
        "next_agent": "context_loader",  # Always start with context loading
        "errors": [],
        "agent_outputs": {}
    }

def validate_state(state: KinnectState) -> tuple[bool, list[str]]:
    """
    Validates that the state object has all required fields.
    Returns (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required string fields
    if not state.get("user_id"):
        errors.append("Missing user_id")
    if not state.get("session_id"):
        errors.append("Missing session_id")
    
    # Check routing is valid
    valid_agents = ["context_loader", "conversational_agent", "memory_extractor", 
                   "diagnostic_analyzer", "alert_handler", "END"]
    if state.get("next_agent") not in valid_agents:
        errors.append(f"Invalid next_agent: {state.get('next_agent')}")
    
    # Check cognitive score is in valid range
    if not (0 <= state.get("cognitive_score", -1) <= 100):
        errors.append(f"Cognitive score out of range: {state.get('cognitive_score')}")
    
    return (len(errors) == 0, errors)