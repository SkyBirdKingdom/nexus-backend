from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.state import AgentState
from app.agents.tools.knowledge_retriever import search_knowledge_base
from app.core.llm_factory import LLMFactory
import json

# Worker 包含两个脑区：一个用来选工具 (正常温度)，一个用来提纯保真 (极低温度)
worker_llm = LLMFactory.get_text_llm(temperature=0.1).bind_tools([search_knowledge_base])
summarize_llm = LLMFactory.get_text_llm(temperature=0.01)

def worker_node(state: AgentState):
    """
    【架构解析：执行器模式 (Executor Pattern)】
    负责调用工具获取外界数据，并进行数据清洗 (提纯)。
    """
    tasks = state.get("tasks", [])

    # 防御性装甲：如果任务数组还是空了，用用户的原话兜底
    current_task = tasks[0] if len(tasks) > 0 else state.get("objective", "全局检索")
    print(f"\n👷 [研究员] 正在执行任务: {current_task}")
    
    messages = [
        SystemMessage(content="""
        你是一个精通信息检索的顶尖研究员。
        你的任务是将用户的意图转化为最有利于搜索引擎（基于向量语义和Reranker交叉编码器）召回的查询语句。
        
        【🚨 核心检索规则（死命令）】：
        1. 绝对禁止输出散乱的关键词或词袋（如 "平台 成果 情况 Results"）！这种格式会严重破坏语义连贯性，导致大模型的向量空间检索失败。
        2. 必须输出一句完整、连贯的自然语言问句或陈述句。
        3. 尽量保留用户原话中的特定名词、项目代号和业务词汇。
        
        例如：
        用户输入: "如何利用自动伸缩提升可靠性" -> 工具调用参数: "如何利用自动伸缩（Auto Scaling）提升系统的可靠性？"
        用户输入: "各平台成果情况" -> 工具调用参数: "各平台的研究成果和具体情况明细是什么？"
        """),
        HumanMessage(content=current_task)
    ]
    
    response = worker_llm.invoke(messages)

    # 安全切片：防止切片报错
    remaining_tasks = tasks[1:] if len(tasks) > 0 else []
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        current_past_steps = state.get("past_steps", [])

        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_knowledge_base":
                print(f"   -> 🔍 触发检索, 参数: {tool_call['args']}")
                runnable_config = {"configurable": {"user_id": state.get("user_id")}}
                tool_result_str = search_knowledge_base.invoke(tool_call["args"], config=runnable_config)
                
                try:
                    # 🚨 核心改造：将检索结果解析为数组，逐条压入记忆库
                    chunks = json.loads(tool_result_str)
                    for chunk in chunks:
                        # 格式：(任务名/标题, 提纯事实, 原始Chunk)
                        # 为了 100% 防患翻译漂移，提纯事实我们直接使用原始 Chunk
                        current_past_steps.append((chunk["title"], chunk["content"], chunk))
                except Exception as e:
                    # 解析失败的保底逻辑
                    current_past_steps.append((current_task, tool_result_str, tool_result_str))
        print("   -> ✅ 原生情报已全部打散并存入全局上下文。")
        return {
            "tasks": remaining_tasks, 
            "past_steps": current_past_steps 
        }
    else:
        # 没有调用工具时的处理
        current_past_steps = state.get("past_steps", [])
        current_past_steps.append((current_task, response.content, response.content))
        return {
            "tasks": remaining_tasks, 
            "past_steps": current_past_steps 
        }
        