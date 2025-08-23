from typing import Annotated, Sequence, TypedDict, Dict, Optional, Literal, Any, List
import asyncio
import json
import os
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from mcp.client.sse import sse_client
from complex_langgraph.query_agent.query_agent import create_query_agent
from complex_langgraph.normalize_agent.agent import create_normalize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from complex_langgraph.a2a_client.a2a_ocr import ocr_client
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode

GOOGLE_API_KEY = "AIzaSyByctz3rxpT0W_s9bEV2lJUyXDDvXvP6Ys"


client = MultiServerMCPClient({
    "storage": {
        "url": "http://127.0.0.1:9000/mcp",
        "transport": "streamable_http",
    },
    "rag": {
        "url": "http://127.0.0.1:9001/mcp",
        "transport": "streamable_http",
    },
    "query": {
        "url": "http://127.0.0.1:9002/mcp",
        "transport": "streamable_http",
    },
})



class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    next: Literal["ocr_node", "normalize_node", "store_node", "query_node", "rag_node", "FINISH"]
    cur_reasoning: str
    errors: List[str]

    receipt_uri: Optional[str]
    receipt_lines: Optional[str]

    query_text: Optional[str]
    query_result: Optional[Dict[str, Any]]

    ocr_text: Optional[str]
    normalized_receipt: Optional[Dict[str, Any]]

    vendor_name: Optional[str]
    receipt_category: Optional[str]
    receipt_date: Optional[str]
    total_amount: Optional[float]

    storage_status: Optional[Literal["ok", "failed"]]
    storage_result: Optional[Dict[str, Any]]

    rag_text: Optional[str]
    rag_result: Optional[Dict[str, Any]]


## structured output for llm to help update states
class RouteDecision(BaseModel):
    next: Literal["ocr_node", "normalize_node", "store_node", "query_node", "rag_node", "FINISH"]
    reasoning: str
    receipt_lines: Optional[str] = None
    query_text: Optional[str] = None
    receipt_uri: Optional[str] = None
    rag_text: Optional[str] = None
    reply: Optional[str] = None  ## this is for general chat non receipt related questions


SUPERVISOR_SYS = SystemMessage(
    content=(
        "You are the supervisor of a receipt system. Return a JSON object with fields:\n"
        "  - next: one of ['ocr_node','normalize_node','store_node','query_node','rag_node','FINISH']\n"
        "  - reasoning: short justification of your choice\n"
        "  - receipt_uri (optional): If the latest user message contains an image URI, set this to that URI.\n"
        "  - receipt_lines (optional): If the user provided raw receipt lines.\n"
        "  - query_text (optional): If the user asked a question about spending or stored receipts.\n"
        "  - rag_text (optional): If the user asked a question about Lebanon’s economy or product prices.\n"
        "\n"
        "Routing rules:\n"
        "1) If there is a normalized receipt in state but it hasn’t been stored yet → next='store_node'.\n"
        "2) If the task is complete (e.g. storage ok or query answered) → next='FINISH'.\n"
        "3) If OCR text or receipt lines are available and there is no normalized receipt yet → next='normalize_node'.\n"
        "4) If there is a receipt URI but no OCR text → next='ocr_node'.\n"
        "5) If the user asked a database query about spending → next='query_node'.\n"
        "6) If the user asked about Lebanon’s economy or prices → next='rag_node'.\n"
        "Never call the same node twice for the same receipt. Follow the sequence OCR → Normalize → Store without repeating steps. Never invent or hallucinate receipt data."
    )
)

def supervisor_node(state: AgentState) -> AgentState:

    """Router/decision maker that chooses the next node and updates the state."""

    msgs = list(state.get("messages", []))
    summary = (
        f"State: has_receipt_uri={state.get('receipt_uri')}, "
        f"has_receipt_lines={state.get('receipt_lines') or state.get('ocr_text')}, "
        f"has_normalized_receipt={state.get('normalized_receipt')}, "

    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
    decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke(
        [SUPERVISOR_SYS, SystemMessage(content=summary), *msgs]
    )

    if decision.reply:
        msg_reply = AIMessage(name="supervisor", content=decision.reply)
    else:
        msg_reply = AIMessage(name="supervisor", content=f"[route={decision.next}] {decision.reasoning}")

 
    state["cur_reasoning"] = decision.reasoning
    state["messages"] = [*msgs, msg_reply]
    state["next"] = decision.next

    if decision.receipt_uri:
        state["receipt_uri"] = decision.receipt_uri.strip()
    if decision.receipt_lines:
        state["receipt_lines"] = decision.receipt_lines.strip()
    if decision.query_text:
        state["query_text"] = decision.query_text.strip()
    if decision.rag_text:
        state["rag_text"] = decision.rag_text.strip()

    return state


async def ocr_node(state: AgentState) -> AgentState:

    """Extract text from a receipt image via the remote OCR agent."""
    msgs = list(state.get("messages"))
    uri = state.get("receipt_uri")

    if not uri:
        state["messages"] = msgs
        state["errors"] = ["No receipt URI."]
        return state

    result = await ocr_client.send_ocr(uri)
    st = (result.get("state")).lower()

    if st == "completed":
        text = (result.get("text")).strip()
        if not text:
            state["messages"] = msgs
            state["errors"] = ["OCR returned empty text."]
            return state

        state["messages"] = [AIMessage(name="ocr_node", content="OCR done.")]
        state["receipt_lines"] = text
        state["ocr_text"] = text
        state["receipt_uri"] = None
        return state

    if st == "input_required":
        state["messages"] = [AIMessage(name="ocr_node", content=result.get("text"))]
        return state

    state["messages"] = [AIMessage(name="ocr_node", content=f"OCR error: {result.get('text')}")]
    state["errors"] = [f"OCR error: {result.get('text')}"]
    return state


def normalize_node(state: AgentState) -> AgentState:

    """Normalize extracted receipt lines into structured data."""
    
    msgs = list(state.get("messages"))
    lines = state.get("receipt_lines")

    if not lines:
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                lines = (m.content or "").strip()
                break

    if not lines:
        state["messages"] = msgs
        state["errors"] = ["No lines to normalize."]
        return state

    agent = create_normalize_agent()
    resp = agent.invoke([HumanMessage(content=lines)])

    vendor = resp.vendor_name
    total = resp.total_amount
    normalized = resp.model_dump()

    state["messages"] = [*msgs, AIMessage(name="normalize_node", content=f"Normalized receipt from {vendor}, total {total}.")]
    state["normalized_receipt"] = normalized
    state["vendor_name"] = vendor
    state["receipt_category"] = resp.receipt_category
    state["receipt_date"] = resp.date
    state["total_amount"] = total
    state["receipt_lines"] = None
    return state


store_prompt = SystemMessage(
    content=(
        "You are a receipt-storage assistant. "
        "When provided with a normalized receipt in JSON, you MUST call "
        "the `store_receipt` tool and pass the JSON as its argument. "
        "Do not answer with natural language; just call the tool."
    )
)


async def store_node(state: AgentState) -> AgentState:
    """Persist a normalized receipt into the database using remote tools."""
    msgs = list(state.get("messages"))
    doc = state.get("normalized_receipt")

    if not doc:
        state["messages"] = msgs
        state["errors"] = ["Nothing to store."]
        return state

    conversation = msgs + [store_prompt, HumanMessage(content=json.dumps(doc))]

    async with client.session("storage") as session:
        tools = await load_mcp_tools(session=session)
        tool_map = {t.name: t for t in tools}

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, api_key=GOOGLE_API_KEY)
        model = llm.bind_tools(list(tool_map.values()))

        ai_msg = await model.ainvoke(conversation)
        conversation.append(ai_msg)

        storage_results = []
        tool_errors = []
        if getattr(ai_msg, "tool_calls", None):
            outputs = []
            for tool_call in ai_msg.tool_calls:
                try:
                    result = await tool_map[tool_call["name"]].ainvoke(tool_call["args"])
                    storage_results.append(result)
                    outputs.append(
                        ToolMessage(
                            content=json.dumps(result),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
                except Exception as e:
                    err = f"Error: {str(e)}. Please re-check your tool usage and try again."
                    tool_errors.append(err)
                    outputs.append(
                        ToolMessage(
                            content=err,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )

            conversation.extend(outputs)
            final_ai = await model.ainvoke(conversation)
            conversation.append(final_ai)
        else:
            final_ai = ai_msg
            tool_errors.append("Model did not call the storage tool.")

    status = "ok" if storage_results and not tool_errors else "failed"
    content = "Stored successfully" if status == "ok" else f"Storage failed: {tool_errors or 'unknown error'}"

    state["messages"] = [*msgs, AIMessage(name="store_node", content=content)]
    state["storage_status"] = status
    state["storage_result"] = storage_results
    state["normalized_receipt"] = None
    return state


async def query_node(state: AgentState) -> AgentState:
    """Answer a query about stored receipts from the database."""
    msgs = list(state.get("messages"))
    q = state.get("query_text")

    if not q:
        state["messages"] = msgs
        state["errors"] = ["No query text."]
        return state

    payload = {"messages": [("user", q)]}

    async with client.session("query") as session:
        tools = await load_mcp_tools(session=session)
        agent = create_query_agent(tools)
        res = await agent.ainvoke(payload)

    answer_text = getattr(res, "content", str(res))

    state["messages"] = [*msgs, AIMessage(name="query_node", content=answer_text)]
    state["query_result"] = {"content": answer_text}
    state["query_text"] = None
    return state

async def rag_node(state: AgentState) -> AgentState:
    """Execute a RAG query for economic questions using the remote RAG agent."""
    msgs: List[BaseMessage] = list(state.get("messages", []))
    q: Optional[str] = state.get("rag_text")

    async with client.session("rag") as session:
        tools = await load_mcp_tools(session=session)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY)
        model = llm.bind_tools(list(tools))
        tool_node = ToolNode(list(tools))

        conversation: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "You are a RAG operator. You MUST call tools to answer. "
                    "You have two options query from an sql database for specific prices in lebanon or articles_retrieval tool when asked about anything related to lebanese economy/inflation"
                    "Never answer directly without calling at least one tool. "
                    "Prefer SQL if the user asks for a price in a specific city. "
                    "Workflow for SQL and prices in Lebanon always do this never skip a step: (1) call list_tables, (2) call tables_schema on tables, "
                    "(3) call execute_sql to fetch the value, (4) answer concisely. The 4 are necessary"
                    "Workflow for being asked about economy in Lebanon or inflation or something related: call articles_retrieval_tool"
                    "Never ask user for table details"
                    "When using execute tool, generate clear, accurate SQL queries, only SELECT queries are allowed. Query on more than one table and compare results"
                    "When querying for an item the item name may vary (e.g., 'Milk' vs 'Milk Powder (Full Fat)'): "
                    "FIRST run: SELECT DISTINCT Item FROM product_prices_in_cities "
                    "WHERE lower(Item) LIKE '%milk%'; "
                    "Then choose the best match and run a precise SELECT."
                    "If SQL querying isnt a success, try articles_retrieval_tool maybe you can find some info there about prices, let this be your last resort "
                    "The tools provided will give you the answer you just take that info and answer using them"
                    "You will form your final answer based on the retrived tool answers only"
                )
            ),
            HumanMessage(content=q),
        ]

        response: AIMessage = await model.ainvoke(conversation)
        conversation.append(response)

        while getattr(response, "tool_calls", None):
            tool_state: Dict[str, Any] = await tool_node.ainvoke({"messages": [response]})
            tool_messages: List[ToolMessage] = tool_state.get("messages", [])
            conversation.extend(tool_messages)
            response = await model.ainvoke(conversation)
            conversation.append(response)

        answer_text: str = getattr(response, "content", str(response))

    state["messages"] = [*msgs, AIMessage(name="rag_node", content=answer_text)]
    state["rag_result"] = {"content": answer_text}
    state["rag_text"] = None
    return state

ROUTES = {"ocr_node", "normalize_node", "store_node", "query_node", "rag_node"}


def router(state: AgentState):
    nxt = state.get("next")
    if nxt == "FINISH":
        return END
    return nxt if nxt in ROUTES else END


graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("ocr_node", ocr_node)
graph.add_node("normalize_node", normalize_node)
graph.add_node("store_node", store_node)
graph.add_node("query_node", query_node)
graph.add_node("rag_node", rag_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    router,
    {
        "ocr_node": "ocr_node",
        "normalize_node": "normalize_node",
        "store_node": "store_node",
        "query_node": "query_node",
        "rag_node": "rag_node",
        END: END,
    },
)

graph.add_edge("ocr_node", "supervisor")
graph.add_edge("normalize_node", "supervisor")
graph.add_edge("store_node", "supervisor")
graph.add_edge("query_node", "supervisor")
graph.add_edge("rag_node", "supervisor")

purr = graph.compile()



##########test
async def run_single_graph_test(prompt: str) -> Dict[str, Any]:
    return await purr.ainvoke({"messages": [("user", prompt)]})


async def main():
    uri_prompt = ("https://www.receiptfaker.com/_next/image?url=https%3A%2F%2Ffirebasestorage.googleapis.com%2Fv0%2Fb%2Freceipt-faker-bbe70.firebasestorage.app%2Fo%2Fimages%252Fk18UG4ClifkR2NmBxaYf%252Fk18UG4ClifkR2NmBxaYf-b92a78e1-69df-487c-9245-5352e16c1fc5.png%3Falt%3Dmedia%26token%3D0ec81385-a86b-49ae-ac25-c2714b1e3d3f&w=640&q=75")
    result = await run_single_graph_test(uri_prompt)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())