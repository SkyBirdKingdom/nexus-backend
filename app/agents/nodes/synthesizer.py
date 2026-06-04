from app.agents.state import AgentState
from app.core.llm_factory import LLMFactory

# 在 Node 内部初始化大模型，做到依赖下沉
llm = LLMFactory.get_text_llm(temperature=0.1)

def synthesizer_node(state: AgentState):
    """
    【架构解析：纯函数 (Pure Function) 思想】
    这个函数接收 State，返回一个字典（State 的更新）。
    它不直接修改全局变量，这在分布式计算和状态机设计中极其重要，保证了测试的可重复性。
    """
    print("\n🧑‍🎨 [主编] 正在接管并生成最终报告...")
    
    past_steps = state.get("past_steps", [])
    # 🚨 读取记忆库传给主编，让它说话连贯
    history = state.get("chat_history", [])
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    
    if not past_steps:
        prompt = f"【历史记忆】:\n{history_str}\n\n用户对你说：{state['objective']}\n请作为 Nexus AI 助手直接简短回复。禁止伪造技术数据。"
    else:
        facts_context = ""
        for task, result in past_steps:
            facts_context += f"【检索线索】: {task}\n【返回情报】: {result}\n\n"
            
        prompt = f"""
        你是 Nexus 系统的首席架构师。
        【历史记忆】:\n{history_str}
        
        基于以上记忆和以下【底层情报】回答用户的【当前指令】。
        【当前指令】: {state['objective']}
        【情报】: \n{facts_context}
        
        【🚨 保真死命令】：包含 `![...](http://...)` 的图片代码必须原封不动输出！
        """
        
    response = llm.invoke(prompt)

    # 🚨 核心逻辑：更新记忆库
    new_history = history.copy()
    new_history.append({"role": "user", "content": state["objective"]})
    new_history.append({"role": "ai", "content": response.content})

    return {
        "final_response": response.content,
        "chat_history": new_history # 将更新后的记忆还给 LangGraph 保存
    }