from typing import List, Tuple
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    【V5.0 记忆强化版状态契约】
    """
    objective: str                        # 当前轮次的用户指令
    user_id: str
    chat_history: List[dict]              # 【新增】长程上下文记忆 (存放历次对话)
    past_steps: List[Tuple[str, str, str]]     # (任务名, LLM提纯后的事实, 纯原始检索Raw Chunk)
    tasks: List[str]                      # 当前轮次的待办任务
    next_node: str                        # 路由方向盘
    final_response: str                   # 最终输出内容