import os, uuid
from typing import Optional, Dict, Any

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard, MessageSendParams, SendMessageRequest,
    SendMessageResponse, Task, TaskState
)

#REFERENCES: "https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/helloworld/test_client.py"
#"https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/airbnb_planner_multiagent/host_agent/routing_agent.py"

class A2AOCR:
    def __init__(self) -> None:
        self._inited = False
        self._card: Optional[AgentCard] = None
        self._url = os.getenv("OCR_AGENT_URL", "http://127.0.0.1:10001")
        self._httpx_client: Optional[httpx.AsyncClient] = None
        self._client: Optional[A2AClient] = None

    async def init(self) -> None:
        if self._inited:
            return
        self._httpx_client = httpx.AsyncClient(timeout=80)
        resolver = A2ACardResolver(httpx_client=self._httpx_client, base_url=self._url)
        card = await resolver.get_agent_card() 
        self._card = card
        self._client = A2AClient(self._httpx_client, card)  
        self._inited = True

    async def send_ocr(self, uri: str) -> Dict[str, Any]:
        """Send the URI to OCR agent"""
        if not self._inited:
            await self.init()

        message_id = uuid.uuid4().hex
        text = (f"image_url: {uri.strip()}")
        payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "messageId": message_id,
            }
        }

        req = SendMessageRequest(id=message_id, params=MessageSendParams(**payload))
        resp = await self._client.send_message(req)

        try:
            task = resp.root.result
        except AttributeError:
            task = resp.result

        out = {
            "state": task.status.state.value,
            "context_id": task.context_id,
            "task_id": task.id,
            "text": None,
        }

        try:
            if task.status.state == TaskState.input_required:
                part = task.status.message.parts[0]
                root = part.root
                out["text"] = root.text
            elif task.status.state == TaskState.completed:
                part = task.artifacts[0].parts[0]
                root = part.root
                out["text"] = root.text
        except Exception:
            pass

        return out

ocr_client = A2AOCR()