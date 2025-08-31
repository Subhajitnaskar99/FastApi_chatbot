import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- FastAPI app + CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY in backend/.env")
client = OpenAI(api_key=api_key)

# --- Schemas ---
class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        allowed = {"system", "user", "assistant"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}")
        return v

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str

# --- Routes ---
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat/", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Ensure there is a system message; add a default if missing
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    if not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {
            "role": "system",
            "content": "You are a helpful, concise assistant."
        })

    # Call OpenAI Chat Completions
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.7,
    )
    reply = resp.choices[0].message.content
    return {"reply": reply}
