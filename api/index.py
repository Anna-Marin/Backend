import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not configured",
        )

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "openrouter/free",
                "messages": [{"role": "user", "content": req.message}],
            },
        )

    try:
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPStatusError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid response from OpenRouter",
        ) from exc

    return {"reply": reply}
