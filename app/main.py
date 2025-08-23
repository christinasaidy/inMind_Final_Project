from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json 
from langchain_core.messages import AIMessage, ToolMessage
from complex_langgraph.graph import purr as graph
from typing import Dict, Any
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from fastapi import HTTPException
import sqlite3

api = FastAPI() 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "databases"
DB_PATH = DB_DIR / "receipts.db"

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")


def _looks_like_dump(s: str) -> bool:
    s = (s or "").lstrip()
    return s.startswith("{'messages':") or s.startswith('{"messages":')

def _is_route_note(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("[route=") and "]" in s

def _extract_final_text(state: Dict[str, Any]) -> str:
    messages = state.get("messages", []) or []
    for m in reversed(messages):
        if isinstance(m, AIMessage):

            if getattr(m, "name"):
                continue
            if getattr(m, "tool_calls"):
                continue
            if isinstance(m.content, str):
                content = m.content.strip()
                if content and not _looks_like_dump(content) and not _is_route_note(content):
                    return content

    for key in ("query_result", "rag_result"):
        v = state.get(key)
        if isinstance(v, dict):
            content = v.get("content")
            if isinstance(content, str) and content.strip():
                if not _looks_like_dump(content):
                    return content.strip()
                else:
                    import re
                    pattern = r"AIMessage\(content='(.*?)',"
                    matches = re.findall(pattern, content, flags=re.DOTALL)
                    if matches:
                        for cand in reversed(matches):
                            cand_clean = (cand or "").strip()
                            if cand_clean and not _is_route_note(cand_clean):
                                return cand_clean

    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str):
            cand = m.content.strip()
            if cand and not _is_route_note(cand):
                return cand
            

class ChatInput(BaseModel):
    messages: list[str]


@api.post("/chat")
async def chat(input: ChatInput):
    msgs = []
    for msg in input.messages:
        msgs.append(("user", msg))

    state = await graph.ainvoke({"messages": msgs})
    final_text = _extract_final_text(state)
    return final_text


@api.get("/receipts/last")
def get_last_receipt():
    if not DB_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Database not found at {DB_PATH}")

    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM receipts ORDER BY rowid DESC LIMIT 1"
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="No rows in 'receipts' table.")

        return dict(row)


if __name__ == "__main__":
    uvicorn.run("app.main:api", host="0.0.0.0", port=8001, reload=True)