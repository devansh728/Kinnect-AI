import chromadb
import uuid
from backend.models import MemoryChunk, MemoryMetadata

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="patient_memories")

def add_memory(user_id: str, content: str, entity_type: str, source: str) -> str:
    memory_id = str(uuid.uuid4())
    metadata = MemoryMetadata(user_id=user_id, entity_type=entity_type, source=source).model_dump()
    
    collection.add(
        documents=[content],
        metadatas=[metadata],
        ids=[memory_id]
    )
    return memory_id

def query_memory(user_id: str, query_text: str, n_results: int = 3) -> list:
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"user_id": user_id} 
    )
    
    extracted = []
    if results['documents'] and results['documents'][0]:
        for doc, meta, doc_id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
            extracted.append({
                "id": doc_id,
                "content": doc,
                "metadata": meta
            })
    return extracted

def delete_memory(memory_id: str):
    collection.delete(ids=[memory_id])
    return True

def update_memory(memory_id: str, new_content: str, new_metadata: dict):
    collection.update(
        ids=[memory_id],
        documents=[new_content],
        metadatas=[new_metadata]
    )
    return True