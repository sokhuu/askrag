import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from bestRAGever import (
    DOCS_PATH,
    EVENT_TEMPLATES,
    PERSIST_DIRECTORY,
    answer_event,
    ask_question,
    build_hybrid_retriever,
    build_or_load_vectorstore,
    classify_event_intent,
)

load_dotenv()

API_KEY = os.environ["RAG_API_KEY"]
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

retriever_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore = build_or_load_vectorstore(DOCS_PATH, PERSIST_DIRECTORY)
    retriever_state["hybrid_retriever"] = build_hybrid_retriever(vectorstore)
    retriever_state["vectorstore"] = vectorstore
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


class ChecklistItem(BaseModel):
    category: str
    obligation: str
    rule: str
    deadline: str
    due_days: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    chat_history: List[ChatMessage]
    event_id: Optional[str] = None
    event_label: Optional[str] = None
    checklist: Optional[str] = None
    items: Optional[List[ChecklistItem]] = None


class EventRequest(BaseModel):
    event_id: str
    context: str = ""


class EventResponse(BaseModel):
    label: str
    checklist: str
    items: List[ChecklistItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(request: AskRequest):
    chat_history = [
        HumanMessage(content=m.content) if m.role == "human" else AIMessage(content=m.content)
        for m in request.chat_history
    ]

    event_id = classify_event_intent(request.question)

    answer = ask_question(
        request.question,
        retriever_state["hybrid_retriever"],
        chat_history,
        vectorstore=retriever_state["vectorstore"],
    )

    updated_history = [
        ChatMessage(role="human" if isinstance(m, HumanMessage) else "ai", content=m.content)
        for m in chat_history
    ]

    if event_id:
        event_label, checklist, items = answer_event(event_id, "", retriever_state["hybrid_retriever"])
        return AskResponse(
            answer=answer,
            chat_history=updated_history,
            event_id=event_id,
            event_label=event_label,
            checklist=checklist,
            items=items,
        )

    return AskResponse(answer=answer, chat_history=updated_history)


@app.post("/event", response_model=EventResponse, dependencies=[Depends(require_api_key)])
def event(request: EventRequest):
    if request.event_id not in EVENT_TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Unknown event_id: {request.event_id}")

    label, checklist, items = answer_event(
        request.event_id,
        request.context,
        retriever_state["hybrid_retriever"],
    )
    return EventResponse(label=label, checklist=checklist, items=items)
