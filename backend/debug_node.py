"""
debug_node.py
Calls the REAL parse_intent_node function used by the graph, with a
state dict shaped exactly like chat_cli.py would pass it. This isolates
whether the bug is in the node's state-handling vs. the raw LLM call.
Run: python debug_node.py
"""
from graph.nodes import parse_intent_node

state = {
    "messages": [],
    "last_location": None,
    "user_message": "is it safe to cycle in Bhopal today?",
}

result = parse_intent_node(state)
print("Full resulting state:")
for k, v in result.items():
    print(f"  {k!r}: {v!r}")