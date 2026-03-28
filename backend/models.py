# backend/models.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class MemoryMetadata(BaseModel):
    user_id: str = Field(..., description="Unique ID of the patient")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    entity_type: str = Field(..., description="Type of memory: 'fact', 'schedule', 'relationship', 'event'")
    source: str = Field(..., description="Where this memory came from: 'transcript_extract', 'manual_entry'")

class MemoryChunk(BaseModel):
    """
    Chunking Strategy for Long Conversations:
    Instead of embedding full transcripts, the Memory Extraction Agent will isolate specific facts 
    (e.g., "Sarah visited on Tuesday") into atomic sentences. Each fact becomes a single Document Chunk 
    (typically under 50 words) to guarantee highly accurate semantic retrieval during the Pre-Call Context Phase.
    """
    id: str = Field(..., description="Unique ID for the memory chunk")
    content: str = Field(..., description="The atomic fact or isolated memory text")
    metadata: MemoryMetadata