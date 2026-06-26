import os
from contextlib import asynccontextmanager
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from bestRAGever import (
    DOCS_PATH,
    PERSIST_DIRECTORY,
    ask_question,
    build_hybrid_retriever,
    build_or_load_vectorstore,
    load_all_documents,
)

load_dotenv()

API_KEY = os.environ["RAG_API_KEY"]
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

retriever_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore = build_or_load_vectorstore(DOCS_PATH, PERSIST_DIRECTORY)
    documents = load_all_documents(vectorstore)
    retriever_state["hybrid_retriever"] = build_hybrid_retriever(vectorstore, documents)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


def require_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ChatMessage(BaseModel):
    role: Literal["human", "ai"]
    content: str


class AskRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage] = []


class AskResponse(BaseModel):
    answer: str
    chat_history: List[ChatMessage]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(request: AskRequest):
    chat_history = [
        HumanMessage(content=m.content) if m.role == "human" else AIMessage(content=m.content)
        for m in request.chat_history
    ]

    answer = ask_question(request.question, retriever_state["hybrid_retriever"], chat_history)

    updated_history = [
        ChatMessage(role="human" if isinstance(m, HumanMessage) else "ai", content=m.content)
        for m in chat_history
    ]
    return AskResponse(answer=answer, chat_history=updated_history)
