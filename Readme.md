# Kinnect AI
A proactive, multi-agent voice application designed to monitor cognitive health and combat isolation using LangGraph, MCP, and ChromaDB.



# Kinnect AI - Project Breakdown & Task Definition

---

## **PHASE 0: Foundation & Environment Setup**

### **Task 0.1: Project Initialization**
- **Description**: Create the base project structure with proper Python virtual environment and Git repository
- **Deliverables**: 
  - Root directory with proper `.gitignore`
  - Python virtual environment (venv)
  - Initial `requirements.txt` with core dependencies (FastAPI, LangGraph, ChromaDB, langchain-google-genai)
  - Basic README.md with project description
- **Success Criteria**: Can activate venv and install packages without errors

### **Task 0.2: API Keys & Configuration Management**
- **Description**: Set up secure configuration management for Gemini API keys and environment variables
- **Deliverables**:
  - `.env.example` file with placeholder variables
  - `config.py` module that loads environment variables
  - Verification script to test Gemini API connection
- **Success Criteria**: Can make a test call to Gemini API successfully

---

## **PHASE 1: Memory Layer (RAG Foundation)**

### **Task 1.1: ChromaDB Local Setup**
- **Description**: Initialize ChromaDB with proper collections and embedding functions, test basic CRUD operations
- **Deliverables**:
  - ChromaDB persistent client initialization
  - Single collection for patient memories with metadata schema
  - Test script to add/retrieve/delete sample memories
- **Success Criteria**: Can store and retrieve text with semantic search working correctly
- **Independence**: Completely standalone - only requires ChromaDB library

### **Task 1.2: Memory Schema Design**
- **Description**: Define the exact structure of how memories are stored (metadata fields, document format, chunking strategy)
- **Deliverables**:
  - Python dataclass/Pydantic model for Memory objects
  - Document on what metadata fields are stored (user_id, timestamp, entity_type, source)
  - Chunking strategy for long conversations
- **Success Criteria**: Clear, documented schema that handles different memory types (facts, conversations, medical info)
- **Independence**: Pure data modeling - no external dependencies

### **Task 1.3: MCP Memory Server Implementation**
- **Description**: Build the Model Context Protocol server that exposes ChromaDB operations as standardized tools
- **Deliverables**:
  - `memory_mcp.py` with functions: `query_memories()`, `save_memories()`, `update_memory()`, `delete_memory()`
  - MCP server configuration following the official protocol spec
  - Test suite to verify each function works independently
- **Success Criteria**: Can call memory functions via MCP protocol, independent of LangGraph
- **Independence**: Only depends on Task 1.1 and 1.2

---

## **PHASE 2: LangGraph State Machine (Agentic Core)**

### **Task 2.1: State Schema Definition**
- **Description**: Design the TypedDict that represents the shared state object passed between all agents
- **Deliverables**:
  - `state.py` with complete `KinnectState` TypedDict
  - Documentation explaining each field's purpose
  - Validation logic for state transitions
- **Success Criteria**: State object can hold all necessary data (user context, messages, diagnostics, routing info)
- **Independence**: Pure Python typing - no external services needed

### **Task 2.2: Single Agent Proof-of-Concept**
- **Description**: Build ONE working agent (Context Agent) that loads memories and generates a summary
- **Deliverables**:
  - `agents.py` with `context_agent_node()` function
  - Test script that runs this single agent with mock state
  - Verification that it calls MCP memory server and updates state
- **Success Criteria**: Agent runs independently, queries memories, returns updated state
- **Independence**: Only depends on Phase 1 (MCP server) and Task 2.1

### **Task 2.3: StateGraph Workflow Construction**
- **Description**: Build the LangGraph workflow with all nodes and edges, but using placeholder agents
- **Deliverables**:
  - `workflow.py` with complete graph structure
  - All nodes defined (can be empty functions initially)
  - Conditional routing logic implemented
  - Test that shows state flows correctly through the graph
- **Success Criteria**: Graph compiles, state moves through nodes in correct order based on conditions
- **Independence**: Can use mock/placeholder agents - focuses on graph logic only

### **Task 2.4: Remaining Agent Implementations**
- **Description**: Build the other 4 agents (Conversational, Memory Extraction, Diagnostic, Alert) following the same pattern as Task 2.2
- **Deliverables**:
  - Complete `agents.py` with all 5 agent node functions
  - Individual test for each agent with mock inputs
  - Prompt engineering document showing each agent's system prompt
- **Success Criteria**: Each agent works independently, can be tested with mock state
- **Independence**: Only depends on Task 2.1 (state schema) and Phase 1 (memory layer)

---

## **PHASE 3: Conversational Interface (Real-Time Audio)**

### **Task 3.1: Text-Based Conversation Loop**
- **Description**: Build a simple CLI that runs the Conversational Agent in a text-only mode
- **Deliverables**:
  - `cli_chat.py` script that accepts text input and returns text responses
  - Integration with the LangGraph workflow
  - Transcript logging to file
- **Success Criteria**: Can have a full conversation via terminal, transcript is saved
- **Independence**: Only depends on Phase 2 (LangGraph agents)

### **Task 3.2: Speech-to-Text Integration**
- **Description**: Add audio input capability using a local STT solution (Whisper or Gemini's native audio)
- **Deliverables**:
  - `audio_handler.py` module with transcription function
  - Test script that transcribes a sample audio file
  - Decision document: Whisper vs Gemini audio input
- **Success Criteria**: Can transcribe audio files accurately with <2 second latency
- **Independence**: Standalone audio processing module

### **Task 3.3: Text-to-Speech Integration**
- **Description**: Add voice output capability using a local TTS solution
- **Deliverables**:
  - TTS function in `audio_handler.py`
  - Test script that generates speech from text
  - Quality comparison between options (gTTS, Piper, ElevenLabs free tier)
- **Success Criteria**: Generates natural-sounding speech suitable for elderly users
- **Independence**: Standalone audio generation module

### **Task 3.4: Full Audio Conversation Loop**
- **Description**: Combine STT and TTS with the Conversational Agent to enable voice-based interaction
- **Deliverables**:
  - Updated `cli_chat.py` that accepts microphone input and plays audio responses
  - Proper turn-taking logic (detect when user stops speaking)
  - Audio transcript saved alongside text transcript
- **Success Criteria**: Can have a natural voice conversation via CLI
- **Independence**: Combines Task 3.1, 3.2, 3.3

---

## **PHASE 4: Backend API Layer (FastAPI)**

### **Task 4.1: Basic FastAPI Server Setup**
- **Description**: Create FastAPI application with health check and basic endpoints
- **Deliverables**:
  - `main.py` with FastAPI app initialization
  - `/health` endpoint
  - `/status` endpoint that returns system info
  - Test using curl/Postman
- **Success Criteria**: Server runs on localhost, responds to HTTP requests
- **Independence**: Pure FastAPI setup - no AI logic needed

### **Task 4.2: WebSocket Connection Handler**
- **Description**: Implement WebSocket endpoint for real-time bidirectional communication
- **Deliverables**:
  - `/ws/{user_id}` WebSocket endpoint
  - Connection manager class to handle multiple clients
  - Test client script to verify connection
- **Success Criteria**: Can establish WebSocket connection, send/receive messages
- **Independence**: Only depends on Task 4.1

### **Task 4.3: Audio Streaming Protocol**
- **Description**: Design and implement the protocol for streaming audio chunks over WebSocket
- **Deliverables**:
  - Protocol specification document (message format, chunk size, encoding)
  - WebSocket handlers for audio chunks
  - Test that streams audio file and reconstructs it correctly
- **Success Criteria**: Audio quality is preserved, latency is acceptable (<500ms)
- **Independence**: Only depends on Task 4.2

### **Task 4.4: LangGraph Integration with WebSocket**
- **Description**: Connect the WebSocket audio stream to the LangGraph Conversational Agent
- **Deliverables**:
  - Endpoint that triggers LangGraph workflow when audio is received
  - Streaming response back to client
  - Proper error handling and connection cleanup
- **Success Criteria**: WebSocket client can have a conversation with the agent
- **Independence**: Combines Phase 2 (LangGraph), Phase 3 (Audio), Task 4.2-4.3

---

## **PHASE 5: Scheduling & Automation**

### **Task 5.1: Celery & Redis Setup**
- **Description**: Install and configure Celery with Redis backend for task queuing
- **Deliverables**:
  - Redis server running locally
  - Celery worker configuration
  - Test task that executes successfully
- **Success Criteria**: Can queue and execute a simple task via Celery
- **Independence**: Standalone infrastructure setup

### **Task 5.2: Scheduled Call Task Implementation**
- **Description**: Create Celery task that initiates a conversation at a specific time
- **Deliverables**:
  - `tasks.py` with `initiate_daily_call(user_id)` function
  - Celery Beat schedule configuration
  - Test that schedules a call for 1 minute in the future
- **Success Criteria**: Call is automatically initiated at the scheduled time
- **Independence**: Only depends on Task 5.1 and Phase 2 (LangGraph)

### **Task 5.3: User Notification System**
- **Description**: Build the mechanism to "ring" the user's device when a call is scheduled
- **Deliverables**:
  - WebSocket push notification to connected clients
  - Client acknowledgment protocol
  - Retry logic if user doesn't answer
- **Success Criteria**: User device receives notification, can accept or reject call
- **Independence**: Combines Task 5.2 and Task 4.2 (WebSocket)

---

## **PHASE 6: Frontend Interface (React PWA)**

### **Task 6.1: React Project Initialization**
- **Description**: Set up React app with necessary dependencies for WebSocket and audio
- **Deliverables**:
  - Create React App or Vite project
  - Install dependencies (socket.io-client or native WebSocket wrapper)
  - Basic app shell with routing
- **Success Criteria**: Dev server runs, shows placeholder UI
- **Independence**: Pure frontend setup

### **Task 6.2: WebSocket Client Implementation**
- **Description**: Build the client-side WebSocket handler that connects to FastAPI backend
- **Deliverables**:
  - WebSocket connection hook/service
  - Connection state management
  - Automatic reconnection logic
- **Success Criteria**: Can connect to backend WebSocket, send/receive test messages
- **Independence**: Only depends on Task 6.1 and Task 4.2 (backend WebSocket)

### **Task 6.3: Audio Capture & Playback UI**
- **Description**: Build the interface for recording user audio and playing agent responses
- **Deliverables**:
  - Microphone access and recording functionality
  - Audio player component
  - Visual feedback for recording/playing states
- **Success Criteria**: Can record audio in browser, play received audio
- **Independence**: Only depends on Task 6.1

### **Task 6.4: Call Interface Design**
- **Description**: Create the "phone call" UI with answer/hang-up functionality
- **Deliverables**:
  - Incoming call screen with ringtone
  - Active call interface with waveform visualization
  - End call button and post-call summary screen
- **Success Criteria**: UI feels like a natural phone call experience
- **Independence**: Only depends on Task 6.1

### **Task 6.5: Full Frontend-Backend Integration**
- **Description**: Connect all frontend components to the backend workflow
- **Deliverables**:
  - Complete call flow (incoming call → conversation → hang up)
  - Audio streaming both ways
  - Error handling and edge cases
- **Success Criteria**: Can have a full conversation through the web interface
- **Independence**: Combines all of Phase 6 and Phase 4

---

## **PHASE 7: Diagnostic & Alerting System**

### **Task 7.1: Baseline Cognitive Profile Creation**
- **Description**: Design the system for establishing a user's normal cognitive baseline
- **Deliverables**:
  - Schema for storing baseline metrics
  - Algorithm to calculate baseline from first N conversations
  - Script to generate synthetic baseline for testing
- **Success Criteria**: System can identify "normal" patterns for a user
- **Independence**: Pure data modeling and algorithm design

### **Task 7.2: Anomaly Detection Algorithm**
- **Description**: Build the Diagnostic Agent's analysis logic with structured output
- **Deliverables**:
  - Prompt engineering for diagnostic analysis
  - Scoring algorithm (0-100 scale)
  - List of anomaly types with detection criteria
- **Success Criteria**: Agent can identify 5+ types of cognitive anomalies with examples
- **Independence**: Only depends on Phase 2 (LangGraph structure)

### **Task 7.3: MCP Alert Server Implementation**
- **Description**: Build the MCP server for sending alerts (email/SMS)
- **Deliverables**:
  - `alert_mcp.py` with SMTP integration
  - Email template for caregiver notifications
  - Test that sends alert to a test email
- **Success Criteria**: Alert emails are sent successfully with diagnostic summary
- **Independence**: Standalone MCP server

### **Task 7.4: Alert Agent Integration**
- **Description**: Connect the Alert Agent to the diagnostic workflow with conditional routing
- **Deliverables**:
  - Complete Alert Agent implementation
  - Conditional edge in LangGraph that triggers on diagnostic score
  - End-to-end test showing alert sent when threshold is crossed
- **Success Criteria**: Alert is only sent when needed, includes relevant diagnostic info
- **Independence**: Combines Task 7.2, 7.3, and Phase 2 (LangGraph)

---

## **PHASE 8: Testing & Optimization**

### **Task 8.1: Unit Tests for Each Agent**
- **Description**: Write comprehensive tests for each LangGraph agent node
- **Deliverables**:
  - pytest suite with 80%+ coverage
  - Mock MCP servers for isolated testing
  - Test fixtures for different user scenarios
- **Success Criteria**: All agents pass tests with various input conditions
- **Independence**: Depends on Phase 2 but can be done incrementally

### **Task 8.2: Integration Tests for Full Workflow**
- **Description**: Test the complete end-to-end flow from call initiation to alert
- **Deliverables**:
  - Test suite that runs full LangGraph workflow
  - Synthetic conversation data for testing
  - Performance benchmarks (latency, token usage)
- **Success Criteria**: Complete workflow executes without errors, meets latency targets
- **Independence**: Depends on all previous phases being complete

### **Task 8.3: Prompt Optimization**
- **Description**: Refine all agent prompts for better accuracy and lower token usage
- **Deliverables**:
  - A/B test results for different prompt versions
  - Final optimized prompts for each agent
  - Token usage analysis and cost estimation
- **Success Criteria**: Reduced token usage by 20%+ while maintaining quality
- **Independence**: Can be done after agents are working

### **Task 8.4: Memory Retrieval Optimization**
- **Description**: Tune ChromaDB embedding and search parameters for better context relevance
- **Deliverables**:
  - Benchmark of different embedding models
  - Optimal chunk size and overlap settings
  - Top-K tuning for memory retrieval
- **Success Criteria**: Context Agent retrieves most relevant memories 90%+ of the time
- **Independence**: Only depends on Phase 1 (Memory Layer)

---

## **PHASE 9: Containerization & Deployment**

### **Task 9.1: Dockerfile Creation**
- **Description**: Create production-ready Dockerfiles for backend and frontend
- **Deliverables**:
  - `backend/Dockerfile` with multi-stage build
  - `frontend/Dockerfile` with optimized build
  - `.dockerignore` files
- **Success Criteria**: Both containers build successfully and run the application
- **Independence**: Only requires project to be working locally first

### **Task 9.2: Docker Compose Orchestration**
- **Description**: Create docker-compose.yml to run all services together
- **Deliverables**:
  - `docker-compose.yml` with backend, frontend, Redis, ChromaDB
  - Volume mounting for persistent data
  - Network configuration for inter-container communication
- **Success Criteria**: Entire application starts with `docker-compose up`
- **Independence**: Only depends on Task 9.1

### **Task 9.3: Environment Configuration**
- **Description**: Properly configure environment variables for containerized deployment
- **Deliverables**:
  - Separate `.env` files for development and production
  - Docker secrets for sensitive data
  - Documentation on required environment variables
- **Success Criteria**: Can deploy to a fresh machine with just env file and docker-compose
- **Independence**: Depends on Task 9.2

---

## **PHASE 10: Documentation & Polish**

### **Task 10.1: API Documentation**
- **Description**: Generate comprehensive API documentation
- **Deliverables**:
  - FastAPI auto-generated Swagger docs
  - Custom documentation for WebSocket protocol
  - MCP server interface documentation
- **Success Criteria**: External developer could integrate with the API using only docs
- **Independence**: Standalone documentation task

### **Task 10.2: User Guide**
- **Description**: Create end-user documentation for caregivers and family members
- **Deliverables**:
  - Setup guide for tablet device
  - How to interpret diagnostic alerts
  - FAQ and troubleshooting
- **Success Criteria**: Non-technical user can set up and use the system
- **Independence**: Standalone documentation task

### **Task 10.3: Developer README**
- **Description**: Comprehensive technical documentation for developers
- **Deliverables**:
  - Architecture diagram
  - Setup instructions (local and Docker)
  - Contribution guidelines
  - Code style guide
- **Success Criteria**: New developer can get system running in <30 minutes
- **Independence**: Standalone documentation task

### **Task 10.4: Demo Data & Showcase**
- **Description**: Create compelling demo scenarios for portfolio/presentation
- **Deliverables**:
  - Synthetic user profiles with realistic data
  - Sample conversation transcripts
  - Video demo of the system in action
- **Success Criteria**: Effectively demonstrates all key features
- **Independence**: Standalone content creation task

---

## **Project Dependency Map Summary**

```
Phase 0 (Foundation) → Everything depends on this
Phase 1 (Memory) → Independent, feeds into Phase 2
Phase 2 (LangGraph) → Depends on Phase 1, feeds into Phase 3-7
Phase 3 (Audio) → Independent, feeds into Phase 4
Phase 4 (API) → Depends on Phase 2+3
Phase 5 (Scheduling) → Depends on Phase 2+4
Phase 6 (Frontend) → Depends on Phase 4
Phase 7 (Diagnostics) → Depends on Phase 2
Phase 8 (Testing) → Depends on all working phases
Phase 9 (Docker) → Depends on everything working locally
Phase 10 (Docs) → Can start anytime, finalizes at end
```

---

**Each phase and task is designed to:**
1. Work independently with minimal dependencies
2. Be testable in isolation
3. Produce a tangible deliverable
4. Have clear success criteria
5. Be resumable if context is lost (each task has enough description to restart)

Which phase would you like to start with?