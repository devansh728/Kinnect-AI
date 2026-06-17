# backend/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
import chromadb
from backend.websocket.connection_manager import connection_manager
from backend.database import collection, add_memory
import asyncio
from datetime import datetime

class CallScheduler:
    """
    Manages daily call schedules for patients.
    Schedules are stored in ChromaDB metadata with entity_type='schedule'
    and run in-process using APScheduler.
    """
    def __init__(self):
        # Initialize AsyncIOScheduler with memory job store
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': MemoryJobStore()},
            timezone='UTC'
        )
        self.db_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.db_client.get_or_create_collection(name="patient_memories")
        self._is_started = False

    async def start(self):
        """Start the scheduler and load all schedules from ChromaDB."""
        if self._is_started:
            return
        
        # Start the scheduler
        self.scheduler.start()
        self._is_started = True
        print("⏰ CallScheduler: APScheduler started.")
        
        # Load existing schedules from database
        await self.load_all_schedules_from_db()

    async def shutdown(self):
        """Shutdown the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
        self._is_started = False
        print("⏰ CallScheduler: APScheduler shutdown.")

    async def initiate_daily_call(self, user_id: str):
        """Job function triggered by scheduler: pushes call notification to patient via WebSocket."""
        print(f"⏰ Scheduler: Triggering scheduled call for user: {user_id}")
        
        message = {
            "type": "incoming_call",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to WebSocket connection
        delivered = await connection_manager.send_to_user(user_id, message)
        if delivered:
            print(f"📞 Call notification successfully pushed to user: {user_id}")
        else:
            print(f"⚠️ User {user_id} is offline. Missed scheduled call.")

    async def load_all_schedules_from_db(self):
        """Queries ChromaDB for all schedule entries and registers them with APScheduler."""
        try:
            # Query documents with entity_type='schedule'
            results = self.collection.get(
                where={"entity_type": "schedule"}
            )
            
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])
            
            print(f"⏰ CallScheduler: Loading {len(ids)} call schedules from ChromaDB...")
            
            for doc_id, meta in zip(ids, metadatas):
                user_id = meta.get("user_id")
                hour = meta.get("hour")
                minute = meta.get("minute")
                
                if user_id is not None and hour is not None and minute is not None:
                    self.register_scheduler_job(user_id, int(hour), int(minute))
                    
        except Exception as e:
            print(f"❌ CallScheduler error loading schedules: {e}")

    def register_scheduler_job(self, user_id: str, hour: int, minute: int):
        """Registers or replaces a cron job in APScheduler for the patient."""
        job_id = f"call_job_{user_id}"
        
        # Remove existing job if it exists to avoid duplicates
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            
        # Add cron job (runs daily at the specified hour and minute)
        self.scheduler.add_job(
            self.initiate_daily_call,
            'cron',
            hour=hour,
            minute=minute,
            args=[user_id],
            id=job_id,
            replace_existing=True
        )
        print(f"⏰ CallScheduler: Registered daily call job for {user_id} at {hour:02d}:{minute:02d} UTC")

    async def set_call_schedule(self, user_id: str, hour: int, minute: int):
        """Saves schedule metadata into ChromaDB and schedules the APScheduler job."""
        job_id = f"call_job_{user_id}"
        
        # 1. Check for existing schedule document in ChromaDB for this user and delete it
        try:
            existing = self.collection.get(
                where={"user_id": user_id}
            )
            existing_ids = []
            for doc_id, meta in zip(existing.get("ids", []), existing.get("metadatas", [])):
                if meta and meta.get("entity_type") == "schedule":
                    existing_ids.append(doc_id)
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                print(f"⏰ CallScheduler: Deleted {len(existing_ids)} old schedule records for {user_id}")
        except Exception as e:
            print(f"⚠️ Warning checking for existing schedule: {e}")
            
        # 2. Add new schedule record to ChromaDB
        metadata = {
            "user_id": user_id,
            "entity_type": "schedule",
            "source": "system",
            "hour": hour,
            "minute": minute
        }
        
        import uuid
        doc_id = f"sched_{uuid.uuid4().hex[:12]}"
        content = f"Daily call scheduled at {hour:02d}:{minute:02d} UTC"
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"⏰ CallScheduler: Saved schedule metadata in ChromaDB for {user_id} (ID: {doc_id})")
        
        # 3. Register with active APScheduler
        self.register_scheduler_job(user_id, hour, minute)

# Global call scheduler instance
call_scheduler = CallScheduler()
