from mcp.server.fastmcp import FastMCP
from backend.database import add_memory, query_memory, delete_memory, update_memory

mcp = FastMCP("KinnectMemoryServer")

@mcp.tool()
def save_patient_memory(user_id: str, content: str, entity_type: str, source: str) -> str:
    """Saves a new, isolated fact or schedule into the patient's long-term memory graph."""
    memory_id = add_memory(user_id, content, entity_type, source)
    return f"Successfully saved memory with ID: {memory_id}"

@mcp.tool()
def fetch_patient_memories(user_id: str, query_text: str, limit: int = 3) -> str:
    """Searches the patient's vector database for memories semantically relevant to the query."""
    results = query_memory(user_id, query_text, limit)
    if not results:
        return "No relevant memories found."
    
    formatted = [f"- {res['content']} (Type: {res['metadata']['entity_type']})" for res in results]
    return "\n".join(formatted)

@mcp.tool()
def remove_patient_memory(memory_id: str) -> str:
    delete_memory(memory_id)
    return f"Deleted memory: {memory_id}"

if __name__ == "__main__":
    mcp.run(transport='stdio')