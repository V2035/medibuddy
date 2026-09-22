"""
debug_graph.py
Runs the actual compiled graph's .invoke() once, printing the full
final state so we can see exactly what happened at every step.
Run: python debug_graph.py
"""
from graph.build_graph import build_graph

app = build_graph()

state = {
    "messages": [],
    "last_location": None,
    "user_message": "is it safe to cycle in Bhopal today?",
}

result = app.invoke(state)

print("Full final state after graph.invoke():")
for k, v in result.items():
    print(f"  {k!r}: {v!r}")