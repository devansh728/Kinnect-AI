# Kinnect AI — Complete Project Description & Architecture

Kinnect AI is a voice-activated, agentic daily check-in and cognitive screening platform for elderly care. The system helps elderly individuals maintain connection and automatically monitors cognitive health indicators (like memory recall, confusion, and behavioral patterns) over time.

---

## 🏗️ System Architecture

The project consists of a **FastAPI backend** running an **in-process scheduler (APScheduler)** and exposing a **WebSocket connection server**. Conversational turns and post-call analyses are managed dynamically using a state-sharing **LangGraph workflow**.

```mermaid
graph TD
    classDef client fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef fastapi fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef langgraph fill:#fff8e1,stroke:#fbc02d,stroke-width:2px;
    classDef database fill:#efebe9,stroke:#5d4037,stroke-width:2px;

    Client[Frontend Client App<br/>Web / Mobile]:::client
    FastAPI[FastAPI Server<br/>main.py]:::fastapi
    SessionManager[Session Manager<br/>In-memory Session Lock]:::fastapi
    Scheduler[APScheduler<br/>Daily Call Scheduler]:::fastapi
    
    subgraph LangGraph Flow
        ContextAgent[1. Context Agent<br/>Pre-Call RAG]:::langgraph
        VoiceLoop[2. Active Call<br/>Whisper STT / Gemini LLM / gTTS]:::langgraph
        MemoryExtractor[3. Memory Extractor<br/>Entity Parser]:::langgraph
        DiagnosticAgent[4. Diagnostic Agent<br/>Cognitive Scoring & Deviation]:::langgraph
        AlertAgent[5. Alert Handler<br/>Caregiver File Stub Notification]:::langgraph
    end
    
    ChromaDB[(ChromaDB<br/>Vector Database)]:::database
    AlertsFolder[(Alerts Folder<br/>sent_email_*.txt)]:::database

    %% Client and Server routing
    Client <-->|WebSocket JSON & Audio| FastAPI
    FastAPI <--> SessionManager
    Scheduler -->|incoming_call push| Client
    
    %% Database connections
    ContextAgent -->|Semantic retrieval| ChromaDB
    MemoryExtractor -->|add_memory| ChromaDB
    DiagnosticAgent -->|compare_to_baseline| ChromaDB
    DiagnosticAgent -->|update_baseline| ChromaDB
    Scheduler -->|Read schedules| ChromaDB
    
    %% Post-call workflow pipeline
    FastAPI -->|1. start_session| ContextAgent
    ContextAgent -->|2. start conversation| VoiceLoop
    Client -->|3. end_session| MemoryExtractor
    MemoryExtractor -->|4. run diagnostics| DiagnosticAgent
    DiagnosticAgent -->|5. Needs Alert?| AlertAgent
    AlertAgent -->|6. Save alert| AlertsFolder
```

---

## ⚙️ Core Modules

### 1. In-Memory Session & Connection Manager (`session_manager.py`, `connection_manager.py`)
Tracks active WebSocket connections and call states by `user_id`. It leverages `asyncio.Lock` to guarantee concurrency safety during read-modify-write state operations.

### 2. WebSocket Audio & Interaction Protocol (`handlers.py`, `audio_streamer.py`)
Provides real-time, low-latency audio processing:
*   **Speech-to-Text (STT):** Decodes incoming base64 WAV chunks and transcribes them using OpenAI Whisper.
*   **Text-to-Speech (TTS):** Generates responses with natural voice intonation via Google TTS (`gtts`), returning them as base64 MP3 chunks.

### 3. LangGraph Post-Call Workflow (`workflow.py`, `agents.py`)
Compiles a structured pipeline executing immediately after a call ends:
1.  **Memory Extraction Agent:** Scans the call transcript for new facts (appointments, health issues, preferences) and inserts them into ChromaDB.
2.  **Diagnostic Analyzer Agent:** Scores cognitive health (0–100) and detects anomalies. It checks current scores against the rolling historical baseline.
3.  **Alert Handler Agent:** Triggers caregiver warnings if the score falls below `60` or drops by `>20` points compared to the baseline.

### 4. Patient Cognitive Baseline Tracker (`baseline.py`)
Stores historical cognitive scores as custom metadata inside ChromaDB under the `baseline` entity type. It manages rolling averages of the patient's last 10 scores to track cognitive decline over time.

### 5. Call Scheduler (`scheduler.py`)
Runs in-process using APScheduler. It stores custom daily call times (e.g. 10:00 AM) in ChromaDB metadata. When triggered, it pushes an `incoming_call` frame over WebSockets to ring the client application.

### 6. Containerization (`Dockerfile`, `docker-compose.yml`)
Bridges all OS-level system libraries (like `ffmpeg`, `portaudio19`, and `libsndfile`) required for raw audio manipulation inside a single standalone Python image.
