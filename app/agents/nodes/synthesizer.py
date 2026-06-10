# app/agents/nodes/synthesizer.py
from app.agents.state import AgentState
from app.core.llm_factory import LLMFactory

llm = LLMFactory.get_text_llm(temperature=0.1)

def synthesizer_node(state: AgentState):
    print("\n🧑‍🎨 [主编] 正在接管并生成最终报告...")
    
    past_steps = state.get("past_steps", [])
    history = state.get("chat_history", [])
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    
    if not past_steps:
        # 🚨 引导词微调：没找到资料时，让大模型优雅诚实地拒绝
        prompt = f"【历史记忆】:\n{history_str}\n\n用户对你说：{state['objective']}\n请极其明确、诚实地告诉用户：在现有的个人知识库沙箱中未能检索到与该问题相关的底层情报，请重新上传相关文档。绝对禁止伪造或编造任何数据。"
    else:
        facts_context = ""
        for idx, step in enumerate(past_steps):
            facts_context += f"【来源 {idx+1}】: {step[0]}\n【底层情报】: {step[1]}\n\n"
            
        prompt = f"""
        你是 Nexus 系统的首席架构师。
        【历史记忆】:\n{history_str}
        
        基于以上记忆和以下【底层情报】回答用户的【当前指令】。
        【当前指令】: {state['objective']}
        【情报】: \n{facts_context}
        
        【🚨 核心输出规范】：
        1. 【引用溯源】：必须使用标准 Markdown 脚注语法。例如句子末尾添加 `[^1]`。
        2. 【格式熔断】：回答正文后立即停止！绝对禁止在文末生成类似 "### 参考资料" 的附录。
        3. 【图片绝对保真】：当你决定在回答中展示情报里的图片时，必须且只能使用 Markdown 内联语法：`![图片说明](完整的http链接)`。
        
        4. 【🚨 企业级表格与 Artifacts 隔离协议】：
           如果【底层情报】中包含表格数据，你必须保留其完整的 HTML 结构（动态计算 rowspan 还原合并单元格）。
           但是，你绝对不能将 HTML 直接暴露在正文中！你必须将其包裹在一个标记为 `html nexus-artifact` 的 Markdown 代码块中！
           
           正确输出格式示范：
           ```html nexus-artifact
           <div class='table-wrapper'>
             <table class='nexus-table'>...</table>
           </div>
           ```
        """
        
    response = llm.invoke(prompt)

    # 🚨 核心升级：组装当时的序列化 sources，捆绑存入历史状态机
    serialized_sources = []
    for i, step in enumerate(past_steps):
        if len(step) > 2 and isinstance(step[2], dict):
            serialized_sources.append({
                "id": str(i+1),
                "title": step[0],
                "chunk": step[2].get("content", step[1]),
                "rerank_score": step[2].get("rerank_score"),
                "rrf_score": step[2].get("rrf_score")
            })

    new_history = history.copy()
    new_history.append({"role": "user", "content": state["objective"]})
    # AI 消息体打包 sources 统一落库 Postgres
    new_history.append({
        "role": "ai", 
        "content": response.content,
        "sources": serialized_sources 
    })

    return {
        "final_response": response.content,
        "chat_history": new_history
    }