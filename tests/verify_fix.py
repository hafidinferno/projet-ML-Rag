
import httpx
import time
import shutil
import os
from pathlib import Path

API_URL = "http://localhost:8000"

def test_pipeline():
    print("=== Starting Validation Pipeline ===")
    
    # 1. Check Health (initially might be 0 indexed)
    try:
        r = httpx.get(f"{API_URL}/health")
        print(f"Initial Health: {r.json()}")
    except Exception as e:
        print(f"API not running? {e}")
        return

    # 2. Trigger Ingestion (expect clean re-index)
    print("\n>>> Triggering /ingest...")
    start = time.time()
    r = httpx.post(f"{API_URL}/ingest", json={"force_reindex": True}, timeout=120)
    print(f"Ingest Status: {r.status_code}")
    if r.status_code != 200:
        print(f"Ingest Failed: {r.text}")
        return
    
    ingest_data = r.json()
    print(f"Ingest Response: {ingest_data}")
    
    if not ingest_data["success"] or ingest_data["documents_processed"] == 0:
        print("FAILED: Ingestion reported failure or 0 documents.")
        return

    # 3. Check Health Again (must show indexed documents)
    print("\n>>> Checking /health after ingestion...")
    r = httpx.get(f"{API_URL}/health")
    health_data = r.json()
    print(f"Health Response: {health_data}")
    
    if health_data["documents_indexed"] == 0:
        print("FAILED: Health check shows 0 documents after ingestion.")
        # Debug: list documents
        docs = httpx.get(f"{API_URL}/documents").json()
        print(f"List Documents: {docs}")
        return
    
    # 4. Test Chat (must find citations)
    print("\n>>> Testing /chat...")
    payload = {
        "user_message": "Oui c'est une fraude, je confirme. J'ai vu un débit de 500 euros sur Amazon.",
        "fraud_confirmed": True,
        "transaction_context": {
            "amount": 500,
            "currency": "EUR",
            "merchant": "Amazon",
            "channel": "online",
            "date": "2026-01-25",
            "country": "FR"
        }
    }
    
    start_chat = time.time()
    r = httpx.post(f"{API_URL}/chat", json=payload, timeout=60)
    print(f"Chat Status: {r.status_code}")
    print(f"Chat Time: {time.time() - start_chat:.2f}s")
    
    if r.status_code != 200:
        print(f"Chat Failed: {r.text}")
        return

    chat_data = r.json()
    agent_resp = chat_data.get("agent_response", {})
    citations = agent_resp.get("citations", [])
    
    print(f"Chat Success: {chat_data['success']}")
    print(f"Citations Found: {len(citations)}")
    
    if len(citations) > 0:
        print("\nSUCCESS: Pipeline verified!")
        print(f"First citation: {citations[0]['doc_id']} (Score: {citations[0]['score']})")
    else:
        print("\nWARNING: Chat worked but no citations found (maybe check retrieval logic again?)")

if __name__ == "__main__":
    # Optional: try to clear vectordb if possible (might be locked by API)
    # db_path = Path("e:/pml2/projet/vectordb")
    # if db_path.exists():
    #     try:
    #         shutil.rmtree(db_path)
    #         print("Cleared ./vectordb")
    #     except Exception as e:
    #         print(f"Could not clear ./vectordb (locked?): {e}")

    test_pipeline()
