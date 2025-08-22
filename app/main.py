# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json 
from langchain_core.messages import AIMessage, ToolMessage
from complex_langgraph.graph import purr as graph
from typing import Dict, Any

api = FastAPI() 


##helper functions from chatgpt to print AI MESSAGE

def _looks_like_dump(s: str) -> bool:
    s = (s or "").lstrip()
    return s.startswith("{'messages':") or s.startswith('{"messages":')

def _is_route_note(s: str) -> bool:
    # e.g., "[route=query_node] ..."
    s = (s or "").strip()
    return s.startswith("[route=") and "]" in s

def _extract_final_text(state: Dict[str, Any]) -> str:
    messages = state.get("messages", []) or []
    for m in reversed(messages):
        if isinstance(m, AIMessage):

            if getattr(m, "name"):
                continue
            # Skip tool-calling messages
            if getattr(m, "tool_calls"):
                continue
            if isinstance(m.content, str):
                content = m.content.strip()
                if content and not _looks_like_dump(content) and not _is_route_note(content):
                    return content

    # 2) Fall back to node results you stored in state
    for key in ("query_result", "rag_result"):
        v = state.get(key)
        if isinstance(v, dict):
            content = v.get("content")
            if isinstance(content, str) and content.strip():
                if not _looks_like_dump(content):
                    # A simple, clean string in query/rag result
                    return content.strip()
                else:
                    # Try to extract the last human-readable AIMessage from the dump
                    import re
                    # Capture the last AIMessage(content='...') snippet
                    # NOTE: this is a best-effort parser for the repr you showed
                    pattern = r"AIMessage\(content='(.*?)',"
                    matches = re.findall(pattern, content, flags=re.DOTALL)
                    if matches:
                        # Prefer the last non-empty, non-route match
                        for cand in reversed(matches):
                            cand_clean = (cand or "").strip()
                            if cand_clean and not _is_route_note(cand_clean):
                                return cand_clean

    # 3) Last resort: any AI message that looks decent and is not a route note
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

