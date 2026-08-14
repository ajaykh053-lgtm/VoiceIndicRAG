import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY", "").strip("\"' ")
sarvam_key = os.getenv("SARVAM_API_KEY", "").strip("\"' ")

print(f"Groq Key Length: {len(groq_key)} | Prefix: {groq_key[:6]}...")
print(f"Sarvam Key Length: {len(sarvam_key)} | Prefix: {sarvam_key[:6]}...")

# 1. Test Groq Live Generation
try:
    from groq import Groq
    client = Groq(api_key=groq_key)
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "नमस्ते, आप कैसे हैं?"}],
        max_tokens=30
    )
    print(f"[SUCCESS] Groq Llama 3.1 Live Response: {res.choices[0].message.content.strip()}")
except Exception as e:
    print(f"[ERROR] Groq API Failed: {e}")

# 2. Test Sarvam API Connectivity
try:
    import httpx
    resp = httpx.get(
        "https://api.sarvam.ai/v1/models",
        headers={"api-subscription-key": sarvam_key},
        timeout=5.0
    )
    print(f"[SUCCESS] Sarvam AI endpoint reachable (Status {resp.status_code})")
except Exception as e:
    print(f"[INFO] Sarvam connection: {e}")
