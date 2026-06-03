# FastAPI 实例初始化，CORS 跨域配置
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.requests import ChatRequest
from app.agents.graph import complex_agent_app
from app.core.config import settings
from app.api.endpoints import chat # 导入 chat 路由模块

# 初始化 FastAPI 网关
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Multi-Agent RAG System API"
)

app.include_router(chat.router, prefix="/api/v1", tags=["Conversation"])

# 【架构解析：跨域资源共享 (CORS)】
# 允许 Vue 3 前端跨域调用这个 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境请替换为前端真实的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def agent_stream_generator(request: ChatRequest):
    """
    【架构解析：双路 SSE 流式生成器】
    这是我们之前引以为傲的核心逻辑：同时拦截 LangGraph 的节点状态 (updates) 和大模型生成的 Tokens (messages)。
    """
    # 注入全局状态参数
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"objective": request.message}
    
    yield f"data: {json.dumps({'status': 'start', 'content': '🚀 指令已确认，Nexus 引擎开始分配算力...'})}\n\n"
    
    # astream 开启双流模式
    async for event_type, event_data in complex_agent_app.astream(inputs, config=config, stream_mode=["updates", "messages"]):
        
        # --- 通道 1：拦截大模型的 Token 实时流 ---
        if event_type == "messages":
            chunk, metadata = event_data
            if chunk.content and metadata.get("langgraph_node") == "synthesizer":
                yield f"data: {json.dumps({'status': 'text', 'content': chunk.content})}\n\n"
                
        # --- 通道 2：拦截节点的状态更新 ---
        elif event_type == "updates":
            for node_name, node_state in event_data.items():
                if node_name == "planner":
                    plan_list = node_state.get("plan", [])
                    yield f"data: {json.dumps({'status': 'planning', 'tasks': plan_list})}\n\n"
                    yield f"data: {json.dumps({'status': 'start', 'content': '👷 研究员已接管任务，正在高频调取知识库...'})}\n\n"
                    
                elif node_name == "worker":
                    past_steps = node_state.get("past_steps", [])
                    if past_steps:
                        latest_task, latest_fact = past_steps[-1]
                        yield f"data: {json.dumps({'status': 'working', 'task_name': latest_task, 'fact': latest_fact})}\n\n"
                        
                elif node_name == "synthesizer":
                    yield f"data: {json.dumps({'status': 'synthesizing', 'content': '✅ 报告生成完毕！'})}\n\n"

    yield "data: [DONE]\n\n"

@app.post("/api/v1/chat")
async def chat_stream(request: ChatRequest):
    """
    【架构解析：标准 HTTP 响应封装】
    将生成器包裹进 StreamingResponse，指定媒体类型为 text/event-stream，
    前端只需用 Fetch API 即可原生地一行一行读取数据。
    """
    return StreamingResponse(
        agent_stream_generator(request),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health_check():
    """K8s 探针/负载均衡器专用的健康检查接口"""
    return {"status": "ok", "version": settings.VERSION}