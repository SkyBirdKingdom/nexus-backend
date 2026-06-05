from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List
from app.agents.state import AgentState
from app.core.llm_factory import LLMFactory

# 总管需要极其严谨和理性的脑区
supervisor_llm = LLMFactory.get_text_llm(temperature=0.01)

class RouteDecision(BaseModel):
    """【路由契约】：强制大模型必须输出这三个字段，且 next_node 只能二选一"""
    reasoning: str = Field(description="判断用户意图并做出路由决定的思考过程")
    next_node: Literal["worker", "synthesizer"] = Field(description="下一步要调用的节点。需要查知识库选 worker，闲聊或直接回答选 synthesizer。")
    tasks: List[str] = Field(default=[], description="如果分配给 worker，请列出具体的检索任务清单；否则为空。")

def supervisor_node(state: AgentState):
    """
    【架构解析：意图拦截与动态分发】
    这是整个 Swarm 架构的心脏。所有请求必须先过 Supervisor。
    """
    print(f"\n🧠 [总管 Supervisor] 正在评估意图: {state['objective']}")
    
    # 【动态闭环】：如果 Worker 已经干完活了（past_steps有值），总管直接将数据推给主编结案
    if state.get("past_steps"):
        return {"next_node": "synthesizer"}

    # 🚨 读取历史记忆（只提取最近 4 条，防止上下文撑爆）
    history = state.get("chat_history", [])
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
    if not history_str:
        history_str = "暂无历史对话。"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        你是 Nexus 系统的核心总管 (Supervisor)。
        
        【最近的对话记忆】：
        {history_str}
        
        【路由规则 - 绝对服从】：
        1. 意图短路 (闲聊模式)：仅当用户进行纯粹的社交寒暄（如“你好”、“谢谢”、“你是谁”）、或者明显的情绪表达时，才能将 next_node 设为 "synthesizer"。
        2. 深度检索 (专业模式)：只要遇到以下情况，必须将 next_node 设为 "worker"，并提取精准的检索词：
           - 包含具体的产品名、架构名、技术术语（如：微服务, Serverless）
           - 包含任何编号、缩写、代码、项目代号（如：COST02-BP03, ASDF-998）
           - 询问具体的数据、操作步骤、对比分析
           
        【🚨 架构师死命令（宁错杀，不放过）】：
        大模型禁止自作聪明！如果你遇到不懂的词，或者【即使你懂，但它像是一个专业术语】，你必须路由给 "worker" 查阅私有知识库！绝对不允许直接使用你的内置知识回答业务问题！
        """),
        ("user", "【当前新指令】: {objective}")
    ])
    
    # 使用强类型约束，拒绝大模型废话
    chain = prompt | supervisor_llm.with_structured_output(RouteDecision)
    decision = chain.invoke({"objective": state["objective"]})
    
    print(f"   -> 💡 决策: 路由至 [{decision.next_node}] | 理由: {decision.reasoning}")
    
    return {
        "next_node": decision.next_node,
        "tasks": decision.tasks
    }