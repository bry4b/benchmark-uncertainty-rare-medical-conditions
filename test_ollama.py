import requests
import json

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:8b",
        "prompt": (
            "Extract symptoms from this sentence and return only JSON: "
            "The patient had abdominal pain, facial swelling, and nausea."
        ),
        "stream": False
    },
    timeout=120
)

response.raise_for_status()
data = response.json()

print(data["response"])