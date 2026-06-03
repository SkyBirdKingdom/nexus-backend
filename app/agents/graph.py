# 组装 Multi-Agent 流程图的入口
from langgraph.graph import StateGraph, START, END
from app.agents.state import PlanExecuteState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.worker import worker_node
from app.agents.nodes.synthesizer import synthesizer_node

# 【条件判断路由】：决定循环是否继续
def route_step(state: PlanExecuteState):
    """如果还有计划未完成，回传给 Worker；否则交给 Synthesizer 结案。"""
    if isinstance(state.get("plan"), list) and len(state["plan"]) > 0:
        return "worker"
    return "synthesizer"

# ==========================================
# 组装 Multi-Agent 图网络 (编排引擎)
# ==========================================
workflow = StateGraph(PlanExecuteState)

# 1. 注册所有的 Node (工位)
workflow.add_node("planner", planner_node)
workflow.add_node("worker", worker_node)
workflow.add_node("synthesizer", synthesizer_node)

# 2. 定义执行流向 (边)
workflow.add_edge(START, "planner")

# 从 Planner 出来后，进行条件判断路由
workflow.add_conditional_edges(
    "planner",
    route_step,
    {"worker": "worker", "synthesizer": "synthesizer"}
)

# Worker 做完一个任务后，继续判断路由 (循环检查清单)
workflow.add_conditional_edges(
    "worker",
    route_step,
    {"worker": "worker", "synthesizer": "synthesizer"}
)

# 主编写完报告后，流程结束
workflow.add_edge("synthesizer", END)

# 3. 编译图网络为可执行应用
# 在企业级中，这里可以通过传入 checkpointer=SqliteSaver() 开启长程记忆
complex_agent_app = workflow.compile()