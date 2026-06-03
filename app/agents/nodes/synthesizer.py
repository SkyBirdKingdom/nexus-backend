from langchain_community.chat_models import ChatOllama
from app.agents.state import PlanExecuteState
from app.core.config import settings

# 在 Node 内部初始化大模型，做到依赖下沉
llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.1)

def synthesizer_node(state: PlanExecuteState):
    """
    【架构解析：纯函数 (Pure Function) 思想】
    这个函数接收 State，返回一个字典（State 的更新）。
    它不直接修改全局变量，这在分布式计算和状态机设计中极其重要，保证了测试的可重复性。
    """
    print("\n🧑‍🎨 [主编] 正在强制接管 Markdown 渲染引擎...")
    
    facts_context = ""
    for task, result in state.get("past_steps", []):
        facts_context += f"【检索线索】: {task}\n【返回情报】: {result}\n\n"
        
    prompt = f"""
    你是 Nexus 系统的首席多模态架构师。
    
    【用户指令】: {state["objective"]}
    
    【底层搜集到的情报】:
    {facts_context}
    
    【🚨 最高级别系统指令 - 必须绝对服从 🚨】：
    1. 你具备直接展示图片的能力！如果情报中包含了类似 `![架构图](http://...)` 的代码，你 **必须** 将其原封不动地复制到回答中。
    2. 如果用户只是要一张图，直接输出图片 Markdown，禁止生成八股文！
    """
    
    response = llm.invoke(prompt)
    
    # 按照约定，我们将最终报告直接放入字典返回
    # 注意：在真实的 LangGraph 中，通常会在 State 里加一个 `final_response: str` 字段
    # 这里为了兼容我们之前的流式推送，我们将最终结果直接作为返回。
    return {"final_response": response.content}