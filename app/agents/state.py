from typing import List, Tuple
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    【V5.0 记忆强化版状态契约】
    """
    objective: str                        # 当前轮次的用户指令
    chat_history: List[dict]              # 【新增】长程上下文记忆 (存放历次对话)
    past_steps: List[Tuple[str, str]]     # 当前轮次的检索事实 (去掉了 operator.add，改为覆盖更新)
    tasks: List[str]                      # 当前轮次的待办任务
    next_node: str                        # 路由方向盘
    final_response: str                   # 最终输出内容