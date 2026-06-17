# backend/graph/baseline.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
import json
import uuid
import chromadb
from backend.database import collection

class BaselineMetrics(BaseModel):
    """Holds cognitive health baseline metrics for a patient."""
    user_id: str
    avg_cognitive_score: float = 80.0 # Default starting average
    score_history: List[float] = Field(default_factory=list)
    conversation_count: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())

def get_baseline(user_id: str) -> BaselineMetrics:
    """Retrieves the baseline metrics for a patient from ChromaDB, returning a default one if not found."""
    try:
        # Search for baseline document for this user
        results = collection.get(
            where={"user_id": user_id}
        )
        
        for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            if meta and meta.get("entity_type") == "baseline":
                data = json.loads(doc)
                return BaselineMetrics(**data)
    except Exception as e:
        print(f"⚠️ Error reading baseline from ChromaDB: {e}")
        
    # Return default baseline if not found or errored
    return BaselineMetrics(user_id=user_id)

def update_baseline(user_id: str, new_score: float) -> BaselineMetrics:
    """Updates the baseline metrics with a new cognitive score, keeping a rolling window of the last 10 scores."""
    baseline = get_baseline(user_id)
    
    # Update score history (keep last 10)
    baseline.score_history.append(new_score)
    if len(baseline.score_history) > 10:
        baseline.score_history = baseline.score_history[-10:]
        
    # Re-calculate average score
    baseline.avg_cognitive_score = sum(baseline.score_history) / len(baseline.score_history)
    baseline.conversation_count += 1
    baseline.last_updated = datetime.now().isoformat()
    
    # Save back to ChromaDB
    try:
        # Delete old baseline document first
        results = collection.get(
            where={"user_id": user_id}
        )
        old_ids = []
        for doc_id, meta in zip(results.get("ids", []), results.get("metadatas", [])):
            if meta and meta.get("entity_type") == "baseline":
                old_ids.append(doc_id)
        if old_ids:
            collection.delete(ids=old_ids)
            
        # Write new baseline document
        doc_id = f"base_{uuid.uuid4().hex[:12]}"
        content = baseline.model_dump_json()
        metadata = {
            "user_id": user_id,
            "entity_type": "baseline",
            "source": "system"
        }
        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"🔬 Baseline updated for {user_id}: Avg = {baseline.avg_cognitive_score:.1f} (history len = {len(baseline.score_history)})")
    except Exception as e:
        print(f"❌ Error saving baseline to ChromaDB: {e}")
        
    return baseline

def compare_to_baseline(user_id: str, current_score: float) -> float:
    """
    Compares the current score to the patient's average baseline.
    Returns the drop in score. Positive value indicates cognitive decline/drop.
    """
    baseline = get_baseline(user_id)
    # If this is the very first check-in (history empty), drop is 0.0
    if not baseline.score_history:
        return 0.0
        
    drop = baseline.avg_cognitive_score - current_score
    return drop
