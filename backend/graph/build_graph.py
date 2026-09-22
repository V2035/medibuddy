r"""
build_graph.py

Wires the nodes from nodes.py into an actual LangGraph StateGraph with
conditional branching. This is the "real graph with branching" the
assignment requires -- not a single prompt-and-response chain.

Graph shape:

    parse_intent -> fetch_weather --(ok)--> match_sop --(matched)--> compose_answer -> END
                                  \                    \--(no_match)--> no_sop -> END
                                   \                    \--(error)-----> match_error -> END
                                    (failed)
                                     v
                               fail_honestly -> END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import BotState
from .nodes import (
    parse_intent_node,
    fetch_weather_node,
    weather_branch,
    match_sop_node,
    sop_branch,
    compose_answer_node,
    no_sop_node,
    fail_honestly_node,
    match_error_node,
)


def build_graph():
    graph = StateGraph(BotState)

    graph.add_node("parse_intent", parse_intent_node)
    graph.add_node("fetch_weather", fetch_weather_node)
    graph.add_node("match_sop", match_sop_node)
    graph.add_node("compose_answer", compose_answer_node)
    graph.add_node("no_sop", no_sop_node)
    graph.add_node("fail_honestly", fail_honestly_node)
    graph.add_node("match_error", match_error_node)

    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "fetch_weather")

    graph.add_conditional_edges(
        "fetch_weather",
        weather_branch,
        {
            "ok": "match_sop",
            "failed": "fail_honestly",
        },
    )

    graph.add_conditional_edges(
        "match_sop",
        sop_branch,
        {
            "matched": "compose_answer",
            "no_match": "no_sop",
            "error": "match_error",
        },
    )

    graph.add_edge("compose_answer", END)
    graph.add_edge("no_sop", END)
    graph.add_edge("fail_honestly", END)
    graph.add_edge("match_error", END)

    return graph.compile()


if __name__ == "__main__":
    # Quick structural check: does the graph compile without errors?
    app = build_graph()
    print("Graph compiled successfully.")
    print("Nodes:", list(app.get_graph().nodes.keys()))
