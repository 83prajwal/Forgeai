import json
import urllib.request

payload = {"prompt": "Build a simple task tracker app with users and tasks"}
req = urllib.request.Request(
    "http://127.0.0.1:5000/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=60) as response:
    print(response.read().decode())
