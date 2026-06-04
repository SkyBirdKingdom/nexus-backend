from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import AgentState
from app.agents.tools.knowledge_retriever import search_knowledge_base
from app.core.llm_factory import LLMFactory

# Worker 包含两个脑区：一个用来选工具 (正常温度)，一个用来提纯保真 (极低温度)
worker_llm = LLMFactory.get_text_llm(temperature=0.1).bind_tools([search_knowledge_base])
summarize_llm = LLMFactory.get_text_llm(temperature=0.01)

def worker_node(state: AgentState):
    """
    【架构解析：执行器模式 (Executor Pattern)】
    负责调用工具获取外界数据，并进行数据清洗 (提纯)。
    """
    current_task = state["tasks"][0]
    print(f"\n👷 [研究员] 正在执行任务: {current_task}")
    
    messages = [
        SystemMessage(content="你是一个精通信息检索的研究员。请准确提取关键词并调用搜索工具完成任务。"), 
        HumanMessage(content=current_task)
    ]
    
    response = worker_llm.invoke(messages)
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_knowledge_base":
                print(f"   -> 🔍 触发检索, 参数: {tool_call['args']}")
                tool_result = search_knowledge_base.invoke(tool_call["args"])
                
                # 多模态保真提纯指令
                summarize_prompt = f"""
                你是一个无情的情报搬运机器。基于以下【检索结果】，提取关于 '{current_task}' 的核心事实。
                【🚨 数据保真死命令】：如果结果中包含 `![...](http://...)` 的图片代码，绝对不能访问或拒绝，必须原封不动、一字不差地复制到你的总结中！
                
                【检索结果】：
                {tool_result}
                """
                
                final_answer = summarize_llm.invoke(summarize_prompt)
                result_text = final_answer.content
    else:
        result_text = response.content
        
    print("   -> ✅ 提纯事实已存入全局上下文。")
    
    # 🚨 手动读取当前状态并追加，而不是依赖 operator.add
    current_past_steps = state.get("past_steps", [])
    current_past_steps.append((current_task, result_text))

    return {
        "tasks": state["tasks"][1:], 
        "past_steps": current_past_steps 
    }