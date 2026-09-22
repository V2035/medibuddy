"""
chat_cli.py

Minimal terminal chat loop to test the LangGraph agent end-to-end before
wiring up the real frontend. Run: python chat_cli.py

Maintains session memory (message history + last resolved location)
across turns within this one run, same as the eventual chat frontend
will need to.
"""
from graph.build_graph import build_graph

def main():
    app = build_graph()

    session_state = {
        "messages": [],
        "last_location": None,
    }

    print("Weather-Advisory Bot (terminal test mode). Type 'quit' to exit.\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ("quit", "exit"):
            break
        if not user_message:
            continue

        session_state["user_message"] = user_message

        result = app.invoke(session_state)

        answer = result.get("final_answer", "(no answer produced)")
        print(f"\nBot: {answer}\n")

        # Carry session memory forward into the next turn
        session_state["messages"] = result.get("messages", []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
        session_state["last_location"] = result.get("last_location")


if __name__ == "__main__":
    main()
