from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import PlanExecuteState, Plan
from app.core.config import settings

# Planner 只需要普通的温度即可，因为它需要一点点发散思维来拆解任务
planner_llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.5)

def planner_node(state: PlanExecuteState):
    """
    【架构解析：智能路由与意图识别 (Intent Recognition)】
    Planner 的唯一职责是将复杂的人类语言转化为结构化的 JSON 任务流。
    """
    print(f"\n👔 [项目经理] 正在分析用户意图: {state['objective']}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        你是一个聪明的任务拆解专家。请根据用户的【指令】，拆解出需要去知识库搜索的关键字步骤。
        【约束】：如果指令是简短追问，你必须补全真实意图。如果只是闲聊，返回空列表 []。
        """),
        ("user", "【指令】: {objective}")
    ])
    
    planner_chain = prompt | planner_llm.with_structured_output(Plan)
    result = planner_chain.invoke({"objective": state["objective"]})
    
    return {"plan": result.steps}