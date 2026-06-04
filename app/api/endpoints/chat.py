from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.requests import ChatRequest
from app.agents.graph import complex_agent_app
# 引入基础拓扑图
from app.agents.graph import workflow
from app.utils.sse_formatter import format_sse
# 🚨 引入官方推荐的异步持久化引擎
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 创建独立的路由组
router = APIRouter()

async def agent_stream_generator(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    # 🚨 核心重构：回合制状态初始化
    # 只传入新指令并清空工作台，但【不传入】chat_history，
    # 这样 LangGraph 会自动从 SQLite 中读取保留的 chat_history！
    inputs = {
        "objective": request.message,
        "past_steps": [], 
        "tasks": [],      
        "next_node": ""
    }
    
    # 看看用上 sse_formatter 后代码多干净！
    yield format_sse("start", content="🚀 指令已确认，Nexus 引擎开始分配算力...")
    
    # 🚨 核心修复：在异步上下文中，安全打开数据库并动态编译 Agent！
    async with AsyncSqliteSaver.from_conn_string("nexus_memory.db") as memory_saver:
        # 挂载记忆引擎并生成可执行的 app
        complex_agent_app = workflow.compile(checkpointer=memory_saver)
        
        # 开始双路流式推送
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
                            
    yield "data: [DONE]\n\n"

@router.post("/chat")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        agent_stream_generator(request),
        media_type="text/event-stream"
    )