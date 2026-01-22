from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import uvicorn
from collections import defaultdict
import asyncio
from contextlib import asynccontextmanager
import threading
from functools import wraps
import os
# from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()


# ==================== Configuration ====================
class Config:
    MAX_HISTORY_LENGTH = 50
    SESSION_TIMEOUT_MINUTES = 60
    CLEANUP_INTERVAL_SECONDS = 300
    MAX_WORKERS = 100  # Maximum concurrent request handlers
    ENABLE_RATE_LIMITING = False  # Set to True to enable per-user rate limiting
    RATE_LIMIT_REQUESTS = 10  # Requests per minute per user

# ==================== Data Models ====================
class Message(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    user_input_text: str = Field(..., min_length=1, description="User's message")

class ChatResponse(BaseModel):
    user_id: str
    response: str
    conversation_length: int
    timestamp: datetime

class SessionInfo(BaseModel):
    user_id: str
    message_count: int
    first_interaction: datetime
    last_interaction: datetime

# ==================== Thread-Safe Session Manager ====================
# ==================== Optimized Session Manager ====================
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List[Message]] = defaultdict(list)
        self.last_activity: Dict[str, datetime] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self._global_lock = threading.Lock()
        
        # Rate limiting
        self.request_counts: Dict[str, List[datetime]] = defaultdict(list)
    
    async def _get_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create lock for user_id efficiently"""
        if user_id not in self.locks:
            async with self._locks_lock:
                if user_id not in self.locks:
                    self.locks[user_id] = asyncio.Lock()
        return self.locks[user_id]
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Optimized rate limiting check"""
        if not Config.ENABLE_RATE_LIMITING:
            return True
        
        lock = await self._get_lock(user_id)
        async with lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            
            # Remove old requests
            self.request_counts[user_id] = [
                req_time for req_time in self.request_counts[user_id]
                if req_time > cutoff
            ]
            
            # Check limit
            if len(self.request_counts[user_id]) >= Config.RATE_LIMIT_REQUESTS:
                return False
            
            # Add current request
            self.request_counts[user_id].append(now)
            return True
    
    async def add_message(self, user_id: str, role: str, content: str):
        """Optimized message addition"""
        lock = await self._get_lock(user_id)
        async with lock:
            msg = Message(role=role, content=content)
            self.sessions[user_id].append(msg)
            self.last_activity[user_id] = datetime.now()
            
            # Trim history if too long
            if len(self.sessions[user_id]) > Config.MAX_HISTORY_LENGTH:
                self.sessions[user_id] = self.sessions[user_id][-Config.MAX_HISTORY_LENGTH:]
    
    async def get_context(self, user_id: str, max_messages: int = 10) -> List[Dict[str, str]]:
        """Optimized context retrieval"""
        lock = await self._get_lock(user_id)
        async with lock:
            history = self.sessions.get(user_id, [])
            recent = history[-max_messages:] if len(history) > max_messages else history
            return [{"role": msg.role, "content": msg.content} for msg in recent]
    
    async def get_history_length(self, user_id: str) -> int:
        """Fast history length check without copying"""
        lock = await self._get_lock(user_id)
        async with lock:
            return len(self.sessions.get(user_id, []))
    
    async def clear_session(self, user_id: str):
        """Clear session"""
        lock = await self._get_lock(user_id)
        async with lock:
            if user_id in self.sessions:
                del self.sessions[user_id]
            if user_id in self.last_activity:
                del self.last_activity[user_id]
            if user_id in self.request_counts:
                del self.request_counts[user_id]
    
    def cleanup_expired_sessions(self) -> List[str]:
        """Identify expired sessions"""
        current_time = datetime.now()
        timeout = timedelta(minutes=Config.SESSION_TIMEOUT_MINUTES)
        
        with self._global_lock:
            expired_users = [
                user_id for user_id, last_time in self.last_activity.items()
                if current_time - last_time > timeout
            ]
        
        return expired_users
    
    async def remove_expired_sessions(self, expired_users: List[str]):
        """Remove expired sessions"""
        for user_id in expired_users:
            await self.clear_session(user_id)
    
    async def get_session_info(self, user_id: str) -> Optional[SessionInfo]:
        """Get session info"""
        lock = await self._get_lock(user_id)
        async with lock:
            if user_id not in self.sessions or not self.sessions[user_id]:
                return None
            
            messages = self.sessions[user_id]
            return SessionInfo(
                user_id=user_id,
                message_count=len(messages),
                first_interaction=messages[0].timestamp,
                last_interaction=messages[-1].timestamp
            )
    
    def get_all_active_sessions(self) -> List[str]:
        """Get all active sessions"""
        with self._global_lock:
            return list(self.sessions.keys())

# ==================== AI Model with Concurrency Support ====================
class AIModel:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(Config.MAX_WORKERS)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    
    async def generate_response(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate AI response with concurrency control
        Replace with actual AI model integration
        """
        async with self.semaphore:
            # Simulate AI processing time
            # await asyncio.sleep(0.05)  # Reduced for better throughput
            try:
            
                history_length = len(conversation_history)

                # simulate llm generate response approximately 30 seconds per request
                await asyncio.sleep(30)
                
                # if history_length == 0:
                #     return f"Hello! You said: '{user_input}'. How can I assist you today?"
                # elif "bye" in user_input.lower() or "goodbye" in user_input.lower():
                #     return "Goodbye! It was nice chatting with you."
                # else:
                #     return f"[Response to: '{user_input[:50]}...' | Context: {history_length} msgs]"
                res = await self.client.responses.create(
                                                    model="gpt-5-nano",
                                                    text={"format": {"type": "text"},
                                                        "verbosity": "low"},
                                                    reasoning={"effort": "minimal"},
                                                    input=user_input,
                                                )

                return res.output_text
            except Exception as e:
                # Fallback response on error
                print(f"OpenAI API Error: {str(e)}")
                return f"I apologize, but I encountered an error processing your request. Please try again."

        
# ==================== Background Tasks ====================
async def cleanup_task(session_manager: SessionManager):
    """Background task for session cleanup"""
    while True:
        await asyncio.sleep(Config.CLEANUP_INTERVAL_SECONDS)
        expired_users = session_manager.cleanup_expired_sessions()
        if expired_users:
            await session_manager.remove_expired_sessions(expired_users)
            print(f"[Cleanup] Removed {len(expired_users)} expired sessions")

# ==================== FastAPI Application ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("="*60)
    print("Starting AI Chatbot API (Optimized for Concurrency)")
    print(f"Max Concurrent Workers: {Config.MAX_WORKERS}")
    print(f"Rate Limiting: {'Enabled' if Config.ENABLE_RATE_LIMITING else 'Disabled'}")
    print("="*60)
    cleanup_task_handle = asyncio.create_task(cleanup_task(session_manager))
    yield
    # Shutdown
    cleanup_task_handle.cancel()
    print("Shutting down AI Chatbot API...")

app = FastAPI(
    title="AI Chatbot API with Concurrent Session Management",
    description="High-performance chatbot API optimized for concurrent requests",
    version="2.0.0",
    lifespan=lifespan
)

# Initialize managers
session_manager = SessionManager()
ai_model = AIModel()

# ==================== Metrics ====================
class Metrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.lock = threading.Lock()
    
    def increment_total(self):
        with self.lock:
            self.total_requests += 1
    
    def increment_success(self, processing_time: float):
        with self.lock:
            self.successful_requests += 1
            self.total_processing_time += processing_time
    
    def increment_failure(self):
        with self.lock:
            self.failed_requests += 1
    
    def get_stats(self):
        with self.lock:
            avg_time = (
                self.total_processing_time / self.successful_requests 
                if self.successful_requests > 0 else 0
            )
            return {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "avg_processing_time_ms": round(avg_time, 2)
            }

metrics = Metrics()

# ==================== API Endpoints ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - optimized for burst performance
    """
    start_time = asyncio.get_event_loop().time()
    metrics.increment_total()
    
    try:
        # Rate limiting check (fast path if disabled)
        if Config.ENABLE_RATE_LIMITING:
            if not await session_manager.check_rate_limit(request.user_id):
                metrics.increment_failure()
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {Config.RATE_LIMIT_REQUESTS} requests per minute."
                )
        
        # Get conversation context
        conversation_history = await session_manager.get_context(request.user_id)
        
        # Add user message (async)
        add_user_msg_task = session_manager.add_message(
            user_id=request.user_id,
            role="user",
            content=request.user_input_text
        )
        
        # Start AI generation immediately (parallel with adding message)
        ai_response_task = ai_model.generate_response(
            user_input=request.user_input_text,
            conversation_history=conversation_history
        )
        
        # Wait for both tasks
        await add_user_msg_task
        ai_response = await ai_response_task
        
        # Add AI response
        await session_manager.add_message(
            user_id=request.user_id,
            role="assistant",
            content=ai_response
        )
        
        # Get conversation length (fast)
        conversation_length = await session_manager.get_history_length(request.user_id)
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        metrics.increment_success(processing_time)
        
        return ChatResponse(
            user_id=request.user_id,
            response=ai_response,
            conversation_length=conversation_length,
            timestamp=datetime.now(),
            processing_time_ms=round(processing_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        metrics.increment_failure()
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/session/{user_id}", response_model=SessionInfo)
async def get_session_info(user_id: str):
    """Get session information"""
    info = await session_manager.get_session_info(user_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"No session found for user: {user_id}")
    return info

@app.get("/history/{user_id}")
async def get_conversation_history(user_id: str, limit: Optional[int] = None):
    """Get conversation history"""
    history = await session_manager.get_history(user_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No history found for user: {user_id}")
    
    if limit:
        history = history[-limit:]
    
    return {
        "user_id": user_id,
        "message_count": len(history),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in history
        ]
    }

@app.delete("/session/{user_id}")
async def clear_user_session(user_id: str):
    """Clear user session"""
    await session_manager.clear_session(user_id)
    return {"message": f"Session cleared for user: {user_id}"}

@app.get("/sessions/active")
async def get_active_sessions():
    """Get active sessions"""
    active_users = session_manager.get_all_active_sessions()
    return {
        "active_session_count": len(active_users),
        "user_ids": active_users
    }

@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    stats = metrics.get_stats()
    return {
        "requests": stats,
        "active_sessions": len(session_manager.get_all_active_sessions()),
        "config": {
            "max_workers": Config.MAX_WORKERS,
            "rate_limiting_enabled": Config.ENABLE_RATE_LIMITING,
            "rate_limit": Config.RATE_LIMIT_REQUESTS if Config.ENABLE_RATE_LIMITING else None
        }
    }

@app.post("/metrics/reset")
async def reset_metrics():
    """Reset metrics counters"""
    global metrics
    metrics = Metrics()
    return {"message": "Metrics reset successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.get_all_active_sessions()),
        "timestamp": datetime.now().isoformat()
    }

# ==================== Run Server ====================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 worker with async concurrency
        log_level="info",
        access_log=False  # Disable access log for better performance
    )