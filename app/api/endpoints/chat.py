from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.requests import ChatRequest
from app.agents.graph import complex_agent_app
from app.utils.sse_formatter import format_sse

# 创建独立的路由组
router = APIRouter()

async def agent_stream_generator(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"objective": request.message}
    
    # 看看用上 sse_formatter 后代码多干净！
    yield format_sse("start", content="🚀 指令已确认，Nexus 引擎开始分配算力...")
    
    async for event_type, event_data in complex_agent_app.astream(inputs, config=config, stream_mode=["updates", "messages"]):
        if event_type == "messages":
            chunk, metadata = event_data
            if chunk.content and metadata.get("langgraph_node") == "synthesizer":
                yield format_sse("text", content=chunk.content)
                
        elif event_type == "updates":
            for node_name, node_state in event_data.items():
                if node_name == "planner":
                    yield format_sse("planning", tasks=node_state.get("plan", []))
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