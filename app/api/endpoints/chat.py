# app/api/endpoints/chat.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.requests import ChatRequest
from app.schemas.responses import BaseResponse
from app.agents.graph import workflow
from app.utils.sse_formatter import format_sse
from app.core.security import get_current_user
from app.core.database import get_db_connection
from app.core.config import settings
from app.core.llm_factory import LLMFactory

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncio

router = APIRouter()
DB_URI = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

def upsert_chat_session(safe_thread_id: str, user_id: str, first_message: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chat_sessions WHERE thread_id = %s", (safe_thread_id,))
            if not cur.fetchone():
                try:
                    llm = LLMFactory.get_text_llm(temperature=0.1)
                    prompt = f"请将下面这句话概括为4到8个字的精简标题，绝对不要使用任何标点符号：\n{first_message}"
                    title = llm.invoke(prompt).content.strip()[:15]
                except:
                    title = "新的知识探索"
                cur.execute(
                    "INSERT INTO chat_sessions (thread_id, user_id, title) VALUES (%s, %s, %s)",
                    (safe_thread_id, user_id, title)
                )
            else:
                cur.execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s", (safe_thread_id,))
        conn.commit()

async def agent_stream_generator(request: ChatRequest, current_user: dict):
    safe_thread_id = f"{current_user['user_id']}:{request.thread_id}"
    config = {"configurable": {"thread_id": safe_thread_id}}
    
    inputs = {
        "objective": request.message,
        "user_id": current_user["user_id"],
        "past_steps": [], 
        "tasks": [],      
        "next_node": ""
    }
    
    yield format_sse("start", content="🚀 身份已核验，Nexus Postgres 记忆体挂载中...")

    asyncio.create_task(asyncio.to_thread(
        upsert_chat_session, safe_thread_id, current_user["user_id"], request.message
    ))
    
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as memory_saver:
        await memory_saver.setup()
        complex_agent_app = workflow.compile(checkpointer=memory_saver)
        
        async for event_type, event_data in complex_agent_app.astream(inputs, config=config, stream_mode=["updates", "messages"]):
            if event_type == "messages":
                chunk, metadata = event_data
                if chunk.content and metadata.get("langgraph_node") == "synthesizer":
                    yield format_sse("text", content=chunk.content)
                    
            elif event_type == "updates":
                for node_name, node_state in event_data.items():
                    if node_name == "supervisor":
                        tasks = node_state.get("tasks", [])
                        next_node = node_state.get("next_node")
                        if next_node == "synthesizer":
                            yield format_sse("start", content="✨ 意图已识别，直接对话模式...")
                        else:
                            yield format_sse("planning", tasks=tasks)
                            yield format_sse("start", content="👷 任务已下发，研究员正在介入...")
                    elif node_name == "worker":
                        past_steps = node_state.get("past_steps", [])
                        if past_steps:
                            yield format_sse("working", task_name=past_steps[-1][0], fact=past_steps[-1][1])
                            sources_payload = []
                            for i, step in enumerate(past_steps):
                                if len(step) > 2 and isinstance(step[2], dict):
                                    sources_payload.append({
                                        "id": str(i+1), 
                                        "title": step[0], 
                                        "chunk": step[2].get("content", step[1]),
                                        "rerank_score": step[2].get("rerank_score"),
                                        "rrf_score": step[2].get("rrf_score")
                                    })
                            yield format_sse("sources", data=sources_payload)
                            
    yield "data: [DONE]\n\n"

@router.post("/chat")
async def chat_stream(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    return StreamingResponse(agent_stream_generator(request, current_user), media_type="text/event-stream")

# 🚨 核心新增：从 LangGraph PostgresSaver 中捞取历史真实对话树
@router.get("/chat/history/{thread_id}", response_model=BaseResponse)
async def get_chat_history(thread_id: str, current_user: dict = Depends(get_current_user)):
    safe_thread_id = f"{current_user['user_id']}:{thread_id}"
    config = {"configurable": {"thread_id": safe_thread_id}}
    
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as memory_saver:
        complex_agent_app = workflow.compile(checkpointer=memory_saver)
        # 🚨 核心修复：异步检查点必须调用 aget_state 而不能是 get_state
        state_snapshot = await complex_agent_app.aget_state(config)
        
        # 提取状态机变量
        chat_history = state_snapshot.values.get("chat_history", []) if state_snapshot and state_snapshot.values else []
        return BaseResponse(data=chat_history)

@router.get("/chat/sessions", response_model=BaseResponse)
async def get_sessions(current_user: dict = Depends(get_current_user)):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thread_id, title, updated_at FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC",
                (current_user["user_id"],)
            )
            rows = cur.fetchall()
            sessions = []
            for r in rows:
                raw_thread_id = r[0].split(":", 1)[-1]
                sessions.append({
                    "id": raw_thread_id,
                    "title": r[1],
                    "date": r[2].strftime("%m-%d %H:%M")
                })
    return BaseResponse(data=sessions)

@router.delete("/chat/sessions/{thread_id}", response_model=BaseResponse)
async def delete_session(thread_id: str, current_user: dict = Depends(get_current_user)):
    safe_thread_id = f"{current_user['user_id']}:{thread_id}"
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE thread_id = %s AND user_id = %s", (safe_thread_id, current_user["user_id"]))
        conn.commit()
    return BaseResponse(message="会话已从个人沙箱中抹除。")