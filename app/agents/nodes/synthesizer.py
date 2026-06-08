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
        # 🚨 适配三维元组，忽略下标 2 的 raw_chunk，只把提纯事实喂给主编
        for idx, step in enumerate(past_steps):
            facts_context += f"【来源 {idx+1}】: {step[0]}\n【底层情报】: {step[1]}\n\n"
            
        prompt = f"""
        你是 Nexus 系统的首席架构师。
        【历史记忆】:\n{history_str}
        
        基于以上记忆和以下【底层情报】回答用户的【当前指令】。
        【当前指令】: {state['objective']}
        【情报】: \n{facts_context}
        
        【🚨 核心输出规范（死命令）】：
        1. 【引用溯源】：必须使用标准 Markdown 脚注语法。例如句子末尾添加 `[^1]`。
        2. 【格式熔断】：回答正文后立即停止！绝对禁止在文末生成类似 "### 参考资料" 的附录。
        3. 【图片绝对保真】：当你决定在回答中展示情报里的图片时，**必须且只能**使用 Markdown 的内联图片语法：`![图片说明](完整的http链接)`。
           - 🚨 致命错误：大模型经常错误地将图片写成参考式链接 `![图片说明][1]`。这是绝对禁止的！
           - 图片的 `http` 链接必须原封不动地输出，决不能将其替换为脚注序号！
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