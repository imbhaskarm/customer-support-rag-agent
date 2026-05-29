from langgraph.graph import StateGraph, START, END

from customer_support_bot.agent.state import AgentState
from customer_support_bot.agent.nodes import (
    retrieve_node,
    grade_node,
    generate_node,
    faithfulness_node,
    fallback_node,
)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("faithfulness", faithfulness_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("generate", "faithfulness")
    graph.add_edge("fallback", END)

    def route_after_grade(state: AgentState) -> str:
        if state.get("error"):
            return "fallback"
        return state.get("route_decision", "fallback")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    def route_after_faithfulness(state: AgentState) -> str:
        if state.get("error"):
            return "fallback"
        return state.get("route_decision", "fallback")

    graph.add_conditional_edges(
        "faithfulness",
        route_after_faithfulness,
        {
            "approved": END,
            "fallback": "fallback",
        },
    )

    return graph.compile()


AGENT = build_graph()
