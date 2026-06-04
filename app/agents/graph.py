from langgraph.graph import StateGraph, START, END
from app.agents.state import AgentState
from app.agents.nodes.supervisor import supervisor_node
from app.agents.nodes.worker import worker_node
from app.agents.nodes.synthesizer import synthesizer_node


# ==========================================
# 构建 V5.0 星型拓扑网络 (Supervisor Pattern)
# ==========================================
workflow = StateGraph(AgentState)

# 1. 注册工作站
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("worker", worker_node)
workflow.add_node("synthesizer", synthesizer_node)

# 2. 所有的请求起点必须接入总管
workflow.add_edge(START, "supervisor")

# 3. 【核心大脑路由】：根据 Supervisor 写入 state 的 next_node 动态决定流向
def route_from_supervisor(state: AgentState):
    """
    动态分发器：
    如果判定为闲聊 -> 返回 "synthesizer"
    如果判定为专业问题 -> 返回 "worker"
    如果 worker 干完活又回到 supervisor -> 返回 "synthesizer"
    """
    return state.get("next_node", "synthesizer")

# 挂载条件分支
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "worker": "worker",
        "synthesizer": "synthesizer"
    }
)

# 4. Worker 干完活后，必须再回传给总管汇报 (形成循环，直到总管认为可以结案)
# 这个闭环设计是未来引入多个 Worker (查库、查网、跑代码) 的基础！
workflow.add_edge("worker", "supervisor")

# 5. 主编写完即流程结束
workflow.add_edge("synthesizer", END)

# 🚨 编译时挂载记忆引擎
complex_agent_app = workflow.compile()