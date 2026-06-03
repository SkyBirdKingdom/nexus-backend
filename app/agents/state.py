# 图网络状态定义 (PlanExecuteState)
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class PlanExecuteState(TypedDict):
    """
    【架构解析：全局状态契约 (Global State Contract)】
    这是 LangGraph 的血液。每一个 Node 收到这个 State，处理后返回更新的片段。
    面试时可以提：我们通过 TypedDict 明确了状态结构，避免了 Python 字典滥用导致的 KeyError。
    """
    objective: str # 用户最初的指令
    plan: List[str] # 待办任务清单
    
    # Annotated[..., operator.add] 极其关键！
    # 它告诉 LangGraph：当 Node 返回 past_steps 时，不要覆盖旧数据，而是追加 (Append) 进去。
    # 这就是 Redux/Vuex 中常说的 "Reducer" 思想。
    past_steps: Annotated[List[Tuple[str, str]], operator.add] 

class Plan(BaseModel):
    """用于强制大模型输出 JSON 结构的 Pydantic 模型"""
    steps: List[str] = Field(description="The sequentially ordered steps to complete the objective.")