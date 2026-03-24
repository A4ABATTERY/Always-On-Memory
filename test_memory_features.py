import os
import sys
import time
import requests
import subprocess

os.environ["MEMORY_DB"] = "test_memory.db"
# Clean up old test db if it exists
if os.path.exists("test_memory.db"):
    os.remove("test_memory.db")

print("Starting memory agent...")
proc = subprocess.Popen([sys.executable, "agent.py", "--port", "9999"])
time.sleep(5)  # wait for startup

try:
    print("Ingesting memory 1...")
    resp1 = requests.post("http://localhost:9999/ingest", json={"text": "Arbi is using Python 3.10 for the project.", "source": "test"})
    print(resp1.json())
    
    print("\nIngesting memory 2 (Evolution)...")
    resp2 = requests.post("http://localhost:9999/ingest", json={"text": "Update: Arbi has upgraded the project to use Python 3.12 exclusively.", "source": "test"})
    print(resp2.json())
    
    print("\nChecking all memories...")
    mems = requests.get("http://localhost:9999/memories")
    print("Count:", mems.json().get('count'))
    for m in mems.json().get('memories', []):
        print(f"ID: {m['id']} | Sector: {len(m['sector'])}chars | Valid_to: {m['valid_to']} | Score: {m.get('composite_score', 0)}")
    
    print("\nTriggering consolidation to see if valid_to gets set...")
    cons = requests.post("http://localhost:9999/consolidate")
    print(cons.json())

    print("\nChecking all memories after consolidation...")
    mems_after = requests.get("http://localhost:9999/memories")
    for m in mems_after.json().get('memories', []):
        print(f"ID: {m['id']} | Sector: {m['sector']} | Valid_to: {m['valid_to']} | Score: {m.get('composite_score', 0)}")
        print("Summary:", m['summary'])

    print("\nQuerying memory...")
    query_resp = requests.get("http://localhost:9999/query?q=What version of Python is Arbi using?")
    print(query_resp.json())

finally:
    proc.terminate()
    proc.wait()
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
