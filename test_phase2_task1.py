# test_phase2_task1.py
from backend.graph.state_utils import create_initial_state, validate_state

def test_state_creation():
    print("Testing state creation...")
    state = create_initial_state(user_id="test_user_123")
    
    print(f"User ID: {state['user_id']}")
    print(f"Session ID: {state['session_id']}")
    print(f"Initial next_agent: {state['next_agent']}")
    print(f"Messages: {state['messages']}")
    
    is_valid, errors = validate_state(state)
    assert is_valid, f"State validation failed: {errors}"
    print("✓ State creation successful\n")

def test_state_mutation():
    print("Testing state mutation...")
    state = create_initial_state(user_id="test_user_456")
    
    # Simulate agent updating state
    state["retrieved_memories"] = [{"content": "Test memory", "metadata": {}}]
    state["next_agent"] = "conversational_agent"
    state["messages"].append({"role": "user", "content": "Hello"})
    
    print(f"Updated next_agent: {state['next_agent']}")
    print(f"Retrieved memories count: {len(state['retrieved_memories'])}")
    print(f"Messages count: {len(state['messages'])}")
    
    is_valid, errors = validate_state(state)
    assert is_valid, f"State validation failed after mutation: {errors}"
    print("✓ State mutation successful\n")

def test_invalid_state():
    print("Testing invalid state detection...")
    state = create_initial_state(user_id="test_user_789")
    state["next_agent"] = "invalid_agent"  # Intentionally break it
    
    is_valid, errors = validate_state(state)
    assert not is_valid, "Should have detected invalid agent"
    print(f"Correctly caught errors: {errors}")
    print("✓ Invalid state detection successful\n")

if __name__ == "__main__":
    test_state_creation()
    test_state_mutation()
    test_invalid_state()
    print("All Task 2.1 tests passed! ✓")
    
    