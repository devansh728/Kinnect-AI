# backend/graph/workflow.py
from langgraph.graph import StateGraph, END
from backend.graph.state import KinnectState
from backend.graph.agents import (
    context_agent_node,
    conversational_agent_node,
    memory_extraction_node,
    diagnostic_agent_node,
    alert_agent_node
)

def route_after_diagnostic(state: KinnectState) -> str:
    """
    CRITICAL ROUTING FUNCTION: This is what makes the system truly agentic.
    The diagnostic agent's output (needs_alert flag) determines the next step.
    
    This is NOT hardcoded because the LLM decides the value of needs_alert
    based on its analysis of the conversation.
    """
    print(f"\n🔀 ROUTING DECISION: needs_alert = {state['needs_alert']}")
    
    if state["needs_alert"]:
        print("   → Routing to Alert Handler")
        return "alert_handler"
    else:
        print("   → Ending workflow (no alert needed)")
        return "END"

def create_kinnect_workflow():
    """
    Builds the complete LangGraph workflow.
    
    Flow:
    1. Context Loader (retrieves memories)
    2. Conversational Agent (has the conversation)
    3. Memory Extractor (saves new facts)
    4. Diagnostic Analyzer (checks cognitive health)
    5. [Conditional] Alert Handler (only if score is concerning)
    
    The key is step 5 - it only runs if the AI decides it's needed.
    """
    
    # Initialize the graph with our state schema
    workflow = StateGraph(KinnectState)
    
    # Add all agent nodes
    print("Building workflow graph...")
    workflow.add_node("context_loader", context_agent_node)
    workflow.add_node("conversational_agent", conversational_agent_node)
    workflow.add_node("memory_extractor", memory_extraction_node)
    workflow.add_node("diagnostic_analyzer", diagnostic_agent_node)
    workflow.add_node("alert_handler", alert_agent_node)
    
    # Define the entry point
    workflow.set_entry_point("context_loader")
    
    # Pre-call flow (always runs in this order)
    workflow.add_edge("context_loader", "conversational_agent")
    
    # Post-call flow (sequential processing)
    workflow.add_edge("conversational_agent", "memory_extractor")
    workflow.add_edge("memory_extractor", "diagnostic_analyzer")
    
    # CONDITIONAL ROUTING - This is the agentic decision point
    workflow.add_conditional_edges(
        "diagnostic_analyzer",
        route_after_diagnostic,  # Function that decides the route
        {
            "alert_handler": "alert_handler",
            "END": END
        }
    )
    
    # Alert handler always ends the workflow
    workflow.add_edge("alert_handler", END)
    
    print("✓ Workflow graph constructed")
    
    # Compile the graph (makes it executable)
    return workflow.compile()