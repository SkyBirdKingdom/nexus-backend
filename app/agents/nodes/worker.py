# app/agents/nodes/worker.py
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import AgentState
from app.agents.tools.knowledge_retriever import search_knowledge_base
from app.core.llm_factory import LLMFactory
import json

worker_llm = LLMFactory.get_text_llm(temperature=0.1).bind_tools([search_knowledge_base])

def worker_node(state: AgentState):
    tasks = state.get("tasks", [])
    current_task = tasks[0] if len(tasks) > 0 else state.get("objective", "全局检索")
    print(f"\n👷 [研究员] 正在执行任务: {current_task}")
    
    messages = [
        SystemMessage(content="""
        你是一个精通信息检索的顶尖研究员。必须输出一句完整、连贯的自然语言问句或陈述句进行检索。
        """),
        HumanMessage(content=current_task)
    ]
    
    response = worker_llm.invoke(messages)
    remaining_tasks = tasks[1:] if len(tasks) > 0 else []
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        current_past_steps = state.get("past_steps", [])

        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_knowledge_base":
                print(f"   -> 🔍 触发检索, 参数: {tool_call['args']}")
                runnable_config = {"configurable": {"user_id": state.get("user_id")}}
                tool_result_str = search_knowledge_base.invoke(tool_call["args"], config=runnable_config)
                
                # 🚨 核心修复：如果是明确的未找到标帜，直接熔断跳过！不塞入过去步骤
                if tool_result_str == "NOT_FOUND":
                    print("   -> 🛑 检索结果低于置信度阈值，执行格式熔断。")
                    continue
                
                try:
                    chunks = json.loads(tool_result_str)
                    for chunk in chunks:
                        current_past_steps.append((chunk["title"], chunk["content"], chunk))
                except Exception as e:
                    current_past_steps.append((current_task, tool_result_str, tool_result_str))
        
        return {
            "tasks": remaining_tasks, 
            "past_steps": current_past_steps 
        }
    else:
        current_past_steps = state.get("past_steps", [])
        current_past_steps.append((current_task, response.content, response.content))
        return {
            "tasks": remaining_tasks, 
            "past_steps": current_past_steps 
        }