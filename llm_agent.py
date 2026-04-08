import requests
from typing import Any
from core_utils import parse_json_object
from datetime import datetime


def get_groq_client(api_key: str) -> dict[str, str]:
    return {"api_key": api_key}

def groq_chat_completion(
    client: dict[str, str],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    print(datetime.now())
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {client['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1500,
            "top_p": 1,
            "stream": False,
        },
        timeout=60,
    )
    
    print(datetime.now())
    if response.status_code >= 400:
        raise RuntimeError(f"Groq API xətası ({response.status_code}): {response.text[:300]}")

    payload = response.json()
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("Groq API cavabında choices boşdur.")

    content = choices[0].get("message", {}).get("content", "")
    return str(content).strip()

def call_groq_json(client: Any, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    content = groq_chat_completion(client, system_prompt, user_prompt, temperature)
    return parse_json_object(content)

def call_groq_text(client: Any, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    return groq_chat_completion(client, system_prompt, user_prompt, temperature)
