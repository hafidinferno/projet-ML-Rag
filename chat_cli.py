import requests

API_URL = "http://localhost:8000/chat"

context = {
    "amount": 149.99,
    "currency": "EUR",
    "merchant": "AMAZON EU",
    "channel": "online",
    "date": "2026-01-20",
    "country": "LU",
}

history = []

print("Chat CLI (tape 'exit' pour quitter)\n")

while True:
    msg = input("Vous: ").strip()
    if msg.lower() in {"exit", "quit"}:
        break

    payload = {
        "user_message": msg,
        "fraud_confirmed": True,
        "transaction_context": context,
        "conversation_history": history,
    }

    r = requests.post(API_URL, json=payload, timeout=488)
    r.raise_for_status()
    data = r.json()

    agent = data.get("agent_response", {})
    print("\nAgent:", agent.get("customer_message", "(vide)"))

    actions = agent.get("actions", [])
    if actions:
        print("\nActions:")
        for a in actions:
            print("-", a)

    # (optionnel) afficher citations
    print("\nCitations:", agent.get("citations", []))

    # Stocker l'historique si ton backend l'utilise
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": agent.get("customer_message", "")})
    print()
