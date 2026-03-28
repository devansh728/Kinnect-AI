# backend/cli_chat.py
"""
Interactive CLI chat interface for Kinnect AI.
Allows real human conversation with the agent via terminal.
"""

from backend.graph.workflow import create_kinnect_workflow
from backend.graph.state_utils import create_initial_state
from backend.database import add_memory
from datetime import datetime
import json
import os

class KinnectCLI:
    """Command-line interface for Kinnect AI conversations."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.workflow = create_kinnect_workflow()
        self.transcript_dir = "transcripts"
        self.ensure_transcript_dir()
    
    def ensure_transcript_dir(self):
        """Create transcripts directory if it doesn't exist."""
        if not os.path.exists(self.transcript_dir):
            os.makedirs(self.transcript_dir)
    
    def save_transcript(self, state: dict):
        """Save conversation transcript to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.transcript_dir}/{self.user_id}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Kinnect AI Conversation Transcript\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"User ID: {self.user_id}\n")
            f.write(f"Session ID: {state.get('session_id', 'N/A')}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 60 + "\n\n")
            
            # Write full transcript
            transcript = state.get('transcript', '')
            if transcript:
                f.write("CONVERSATION:\n")
                f.write("-" * 60 + "\n")
                f.write(transcript)
                f.write("\n" + "-" * 60 + "\n\n")
            
            # Write diagnostic summary
            f.write("DIAGNOSTIC SUMMARY:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Cognitive Score: {state.get('cognitive_score', 'N/A')}/100\n")
            f.write(f"Anomalies Detected: {len(state.get('anomalies_detected', []))}\n")
            
            if state.get('anomalies_detected'):
                f.write("\nAnomalies:\n")
                for anomaly in state['anomalies_detected']:
                    f.write(f"  - {anomaly}\n")
            
            f.write("\n" + "-" * 60 + "\n\n")
            
            # Write new memories
            new_entities = state.get('new_entities', [])
            if new_entities:
                f.write("NEW MEMORIES EXTRACTED:\n")
                f.write("-" * 60 + "\n")
                for entity in new_entities:
                    f.write(f"  [{entity.get('entity_type', 'unknown')}] {entity.get('content', 'N/A')}\n")
        
        print(f"\n💾 Transcript saved to: {filename}")
        return filename
    
    def run_interactive_session(self):
        """Run an interactive text-based conversation."""
        print("\n" + "=" * 70)
        print("KINNECT AI - Interactive Text Chat")
        print("=" * 70)
        print(f"User: {self.user_id}")
        print("Type your messages below. Type 'quit', 'exit', or 'bye' to end.\n")
        
        # Initialize state
        state = create_initial_state(user_id=self.user_id)
        
        # Step 1: Run Context Agent (pre-call preparation)
        print("🔄 Loading patient context...")
        for step_output in self.workflow.stream(state):
            for node_name, node_output in step_output.items():
                if node_name == "context_loader":
                    state = {**state, **node_output}
                    print(f"✓ Context loaded: {len(state.get('retrieved_memories', []))} memories retrieved\n")
                    break
            break  # Only run context loader
        
        # Step 2: Interactive conversation
        messages = []
        full_transcript = ""
        
        # Get initial greeting from agent
        print("🤖 Agent is calling...\n")
        context_summary = state.get('context_summary', '')
        memories_formatted = "\n".join([
            f"- {m['content']}" for m in state.get('retrieved_memories', [])
        ]) if state.get('retrieved_memories') else "No prior memories."
        
        # Import agent logic
        from backend.graph.agents import llm
        from backend.graph.prompts import CONVERSATIONAL_SYSTEM_PROMPT
        from langchain_core.messages import HumanMessage, AIMessage
        
        system_prompt = CONVERSATIONAL_SYSTEM_PROMPT.format(
            user_id=self.user_id,
            context_summary=context_summary,
            memories_formatted=memories_formatted,
            current_date=datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
        )
        
        # Get agent's initial greeting
        initial_prompt = f"{system_prompt}\n\nStart the daily check-in call. Greet the patient warmly."
        initial_response = llm.invoke([HumanMessage(content=initial_prompt)])
        agent_message = initial_response.content
        
        messages.append({"role": "assistant", "content": agent_message})
        full_transcript += f"Agent: {agent_message}\n"
        
        print(f"🤖 Agent: {agent_message}\n")
        
        # Interactive loop
        turn_count = 0
        while True:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye', 'end']:
                print("\n📞 Ending call...\n")
                break
            
            # Add to transcript
            messages.append({"role": "user", "content": user_input})
            full_transcript += f"User: {user_input}\n"
            turn_count += 1
            
            # Get agent response
            conversation_context = f"{system_prompt}\n\nConversation so far:\n{full_transcript}\n\nRespond naturally to the user's last message."
            response = llm.invoke([HumanMessage(content=conversation_context)])
            agent_message = response.content
            
            messages.append({"role": "assistant", "content": agent_message})
            full_transcript += f"Agent: {agent_message}\n"
            
            print(f"\n🤖 Agent: {agent_message}\n")
        
        # Step 3: Update state with conversation results
        state['messages'] = messages
        state['transcript'] = full_transcript
        state['next_agent'] = 'memory_extractor'
        
        print("🔄 Processing conversation...\n")
        
        # Step 4: Run post-call agents (Memory Extractor, Diagnostic, Alert)
        # Continue the workflow from memory extraction
        final_state = state
        for step_output in self.workflow.stream(state):
            for node_name, node_output in step_output.items():
                final_state = {**final_state, **node_output}
                
                if node_name == "memory_extractor":
                    print(f"✓ Memory extraction complete: {len(node_output.get('new_entities', []))} new facts")
                elif node_name == "diagnostic_analyzer":
                    print(f"✓ Diagnostic analysis complete: Score {node_output.get('cognitive_score', 'N/A')}/100")
                elif node_name == "alert_handler":
                    print(f"✓ Alert sent to caregiver")
        
        # Step 5: Save transcript
        print("\n" + "=" * 70)
        print("CONVERSATION SUMMARY")
        print("=" * 70)
        print(f"Duration: {turn_count} user turns")
        print(f"Transcript length: {len(full_transcript)} characters")
        print(f"New memories: {len(final_state.get('new_entities', []))}")
        print(f"Cognitive score: {final_state.get('cognitive_score', 'N/A')}/100")
        print(f"Alert triggered: {final_state.get('needs_alert', False)}")
        
        self.save_transcript(final_state)
        
        return final_state


def main():
    """Main entry point for CLI chat."""
    import sys
    
    # Get user ID from command line or use default
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = input("Enter patient ID (or press Enter for 'test_patient'): ").strip()
        if not user_id:
            user_id = "test_patient"
    
    # Create and run CLI
    cli = KinnectCLI(user_id=user_id)
    cli.run_interactive_session()


if __name__ == "__main__":
    main()