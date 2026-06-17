# backend/websocket/handlers.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import json
import asyncio
from backend.websocket.connection_manager import connection_manager
from backend.session_manager import session_manager
from backend.websocket.audio_streamer import transcribe_from_base64, synthesize_to_base64
from backend.graph.workflow import create_kinnect_workflow, create_postcall_workflow
from backend.graph.prompts import CONVERSATIONAL_SYSTEM_PROMPT
from backend.graph.agents import llm
from langchain_core.messages import HumanMessage
import os

router = APIRouter()

# Compile the workflows once
workflow = create_kinnect_workflow()
postcall_workflow = create_postcall_workflow()

async def run_postcall_processing(user_id: str, session_id: str, state: dict):
    """Runs memory extraction, diagnostic analyzer, and alert handler agents in the background."""
    print(f"🔄 Starting background post-call processing for {session_id}...")
    try:
        final_state = state
        # Run postcall stream
        # Since LangGraph is synchronous, we run it in a separate thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        
        def run_sync_workflow():
            temp_state = state.copy()
            for step_output in postcall_workflow.stream(temp_state):
                for node_name, node_output in step_output.items():
                    temp_state = {**temp_state, **node_output}
            return temp_state

        final_state = await loop.run_in_executor(None, run_sync_workflow)
        
        new_entities = final_state.get('new_entities', [])
        diagnostic_report = {
            "cognitive_score": final_state.get('cognitive_score', 'N/A'),
            "needs_alert": final_state.get('needs_alert', False),
            "summary": final_state.get('diagnostic_report', {}).get('summary', 'N/A') if isinstance(final_state.get('diagnostic_report'), dict) else str(final_state.get('diagnostic_report', 'N/A')),
            "anomalies": final_state.get('anomalies_detected', [])
        }
        
        # Save transcript to file (reused from cli_chat/voice_chat concept)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("transcripts", exist_ok=True)
        filename = f"transcripts/ws_{user_id}_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("KINNECT AI - WebSocket Session Transcript\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Patient ID: {user_id}\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("-" * 70 + "\n")
            f.write("CONVERSATION:\n")
            f.write("-" * 70 + "\n")
            f.write(final_state.get('transcript', ''))
            f.write("\n" + "-" * 70 + "\n\n")
            f.write("DIAGNOSTIC SUMMARY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Cognitive Score: {diagnostic_report.get('cognitive_score')}/100\n")
            f.write(f"Alert Triggered: {diagnostic_report.get('needs_alert')}\n")
            f.write(f"Summary: {diagnostic_report.get('summary')}\n")
            
        print(f"💾 Post-call processing complete for {session_id}. Transcript saved to {filename}")
        return diagnostic_report, len(new_entities)
    except Exception as e:
        print(f"❌ Error during post-call processing: {e}")
        return {"error": str(e)}, 0

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket check-in session handler.
    No auth required as per requirements.
    """
    await connection_manager.connect(websocket, user_id)
    
    # Get AudioHandler from app state
    audio_handler = websocket.app.state.audio_handler
    
    # Track current session
    current_session_id = None
    system_prompt = ""
    full_transcript = ""
    messages = []
    
    try:
        while True:
            # Wait for text/json message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "start_session":
                if current_session_id:
                    await websocket.send_json({"type": "error", "message": "Session already active"})
                    continue
                
                # Initialize session state
                state = await session_manager.create_session(user_id)
                current_session_id = state["session_id"]
                
                await websocket.send_json({
                    "type": "session_started",
                    "session_id": current_session_id
                })
                
                # Run the Context Loader (retrieve memories) in an executor
                await websocket.send_json({"type": "processing"})
                
                def load_context_sync():
                    temp_state = state.copy()
                    for step_output in workflow.stream(temp_state):
                        for node_name, node_output in step_output.items():
                            if node_name == "context_loader":
                                return {**temp_state, **node_output}
                        break
                    return temp_state
                
                loop = asyncio.get_running_loop()
                state = await loop.run_in_executor(None, load_context_sync)
                await session_manager.update_session(current_session_id, state)
                
                # Notify context loaded
                memories = state.get("retrieved_memories", [])
                context_summary = state.get("context_summary", "")
                await websocket.send_json({
                    "type": "context_loaded",
                    "memories_count": len(memories),
                    "summary": context_summary
                })
                
                # Build system prompt for this session
                memories_formatted = "\n".join([
                    f"- {m['content']}" for m in memories
                ]) if memories else "No prior memories."
                
                system_prompt = CONVERSATIONAL_SYSTEM_PROMPT.format(
                    user_id=user_id,
                    context_summary=context_summary,
                    memories_formatted=memories_formatted,
                    current_date=datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')
                )
                
                # Generate initial agent greeting
                initial_prompt = f"{system_prompt}\n\nStart the daily check-in call. Greet the patient warmly."
                
                # Invoke LLM in executor
                llm_response = await loop.run_in_executor(
                    None, 
                    lambda: llm.invoke([HumanMessage(content=initial_prompt)])
                )
                agent_msg = llm_response.content
                
                messages.append({"role": "assistant", "content": agent_msg})
                full_transcript += f"Agent: {agent_msg}\n"
                
                # Update state
                await session_manager.update_session(current_session_id, {
                    "messages": [{"role": "assistant", "content": agent_msg}],
                    "transcript": full_transcript
                })
                
                # Synthesize greeting
                b64_audio = await loop.run_in_executor(
                    None,
                    lambda: synthesize_to_base64(agent_msg, audio_handler, method="gtts")
                )
                
                await websocket.send_json({
                    "type": "agent_message",
                    "text": agent_msg,
                    "audio": b64_audio
                })
                
            elif msg_type == "user_message" or msg_type == "audio_chunk":
                if not current_session_id:
                    await websocket.send_json({"type": "error", "message": "No active session. Send start_session first."})
                    continue
                
                user_text = ""
                loop = asyncio.get_running_loop()
                
                if msg_type == "audio_chunk":
                    # Transcribe base64 audio chunk
                    b64_data = message.get("data")
                    if not b64_data:
                        await websocket.send_json({"type": "error", "message": "Missing audio data"})
                        continue
                    
                    await websocket.send_json({"type": "processing"})
                    user_text = await loop.run_in_executor(
                        None,
                        lambda: transcribe_from_base64(b64_data, audio_handler)
                    )
                else:
                    user_text = message.get("text", "").strip()
                
                if not user_text:
                    # Whisper couldn't understand or empty text
                    # We can synthesize a friendly prompt
                    prompt_msg = "I'm sorry, I couldn't hear you clearly. Could you repeat that?"
                    b64_audio = await loop.run_in_executor(
                        None,
                        lambda: synthesize_to_base64(prompt_msg, audio_handler, method="gtts")
                    )
                    await websocket.send_json({
                        "type": "agent_message",
                        "text": prompt_msg,
                        "audio": b64_audio
                    })
                    continue
                
                # Update conversation log
                messages.append({"role": "user", "content": user_text})
                full_transcript += f"User: {user_text}\n"
                await session_manager.update_session(current_session_id, {
                    "messages": [{"role": "user", "content": user_text}],
                    "transcript": full_transcript
                })
                
                # Generate agent response
                await websocket.send_json({"type": "processing"})
                conv_context = f"{system_prompt}\n\nConversation so far:\n{full_transcript}\n\nRespond naturally to the user's last message."
                
                llm_response = await loop.run_in_executor(
                    None,
                    lambda: llm.invoke([HumanMessage(content=conv_context)])
                )
                agent_msg = llm_response.content
                
                messages.append({"role": "assistant", "content": agent_msg})
                full_transcript += f"Agent: {agent_msg}\n"
                await session_manager.update_session(current_session_id, {
                    "messages": [{"role": "assistant", "content": agent_msg}],
                    "transcript": full_transcript
                })
                
                # Synthesize agent response
                b64_audio = await loop.run_in_executor(
                    None,
                    lambda: synthesize_to_base64(agent_msg, audio_handler, method="gtts")
                )
                
                await websocket.send_json({
                    "type": "agent_message",
                    "text": agent_msg,
                    "audio": b64_audio
                })
                
            elif msg_type == "end_session":
                if not current_session_id:
                    await websocket.send_json({"type": "error", "message": "No active session"})
                    continue
                
                await websocket.send_json({"type": "processing"})
                
                # Retrieve final session state
                state = await session_manager.end_session(current_session_id)
                
                # Run post-call processing
                diagnostic_report, new_memories_count = await run_postcall_processing(user_id, current_session_id, state)
                
                await websocket.send_json({
                    "type": "session_ended",
                    "diagnostic": diagnostic_report,
                    "new_memories": new_memories_count
                })
                
                current_session_id = None
                
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected abruptly for user: {user_id}")
    except Exception as e:
        print(f"❌ Error in WebSocket handler: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        await connection_manager.disconnect(websocket, user_id)
        
        # If the websocket disconnected abruptly but has an active session,
        # run the post-call workflow to process what we have.
        if current_session_id:
            state = await session_manager.end_session(current_session_id)
            if state:
                # Run in background task to avoid blocking the event loop
                asyncio.create_task(run_postcall_processing(user_id, current_session_id, state))
