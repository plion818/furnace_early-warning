import requests
import sys

API_URL = "http://127.0.0.1:8000"

try:
    print(f"Testing connection to {API_URL}...")
    response = requests.get(f"{API_URL}/docs", timeout=5)
    if response.status_code == 200:
        print("✅ Backend is reachable!")
    else:
        print(f"⚠️ Backend is reachable but returned status code: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to the backend. Is it running?")
    print("Please make sure you have started the backend in a separate terminal using:")
    print("uvicorn main:app --reload")
    sys.exit(1)
except Exception as e:
    print(f"❌ An error occurred: {e}")
    sys.exit(1)
