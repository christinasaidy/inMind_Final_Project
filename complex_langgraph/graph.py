from typing import Annotated, Sequence, TypedDict, Dict, Optional, Literal, Any, List
import asyncio
import json
import os
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

from complex_langgraph.query_agent.query_agent import create_query_agent
from complex_langgraph.normalize_agent.agent import create_normalize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from complex_langgraph.a2a_client.a2a_ocr import ocr_client
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode

from pathlib import Path

import json

GOOGLE_API_KEY= "AIzaSyByctz3rxpT0W_s9bEV2lJUyXDDvXvP6Ys"

SERVER_PARAMS_STORAGE = StdioServerParameters(
    command="python",
    args=[os.path.abspath(r"C:/Users/saidy/OneDrive/Desktop/Smart_Receipt_Assistant/complex_langgraph/store_agent/mcp_server.py")],
)
SERVER_PARAMS_QUERY = StdioServerParameters(
    command="python",
    args=[os.path.abspath(r"C:/Users/saidy/OneDrive/Desktop/Smart_Receipt_Assistant/complex_langgraph/query_agent/mcp_server.py")],
)

SERVER_PARAMS_RAG= StdioServerParameters(
    command="python",
    args=[os.path.abspath(r"C:/Users/saidy/OneDrive/Desktop/Smart_Receipt_Assistant/complex_langgraph/rag/mcp_server.py")],
)


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

class RouteDecision(BaseModel):
    next: Literal["ocr_node", "normalize_node", "store_node", "query_node", "rag_node","FINISH"]
    reasoning: str
    receipt_lines: Optional[str] = None
    query_text: Optional[str] = None
    receipt_uri: Optional[str] = None
    rag_text: Optional[str] = None
    reply: Optional[str] = None 

SUPERVISOR_SYS = SystemMessage(
    content=(
        "You are the supervisor of a receipt system. Return a JSON object with fields:/n"
        "  -next: one of ['ocr_node','normalize_node','store_node','query_node','rag_node','FINISH']/n"
        "  - reasoning: short justification of your choice/n"
        "  - receipt_uri (optional): If the latest user message contains an image URI, set this to that URI./n"
        "  -receipt_lines (optional): If the user provided raw receipt lines./n"
        "  - query_text (optional): If the user asked a question about spending or stored receipts or how much they spent on something./n/n"
        "  - rag_text (optional): If the user asked a question about anything economy of lebanon related and specific prices of products in lebanon and lebanese cities./n/n"
        "Routing rules:/n"
        "1) If there is a normalized JSON but it hasn’t been stored → next='store_node' and this node you must call store_receipt with no exceptions./n"
        "2) If the task is complete (e.g., query answered or storage ok) → next='FINISH'./n"
        "3) Otherwise let your decision depend on the state and the latest user input./n"
        "Never invent data"
        "4)never make up receipt data, upon a given uri you must go to ocr -> normalize -> store and call their tools, no exceptions"
        "If the user inqueries arent Lebanon related or receipt or price related, you may answer them yourself and then ->next ='FINISH' "
    )
)

##supervisor node
def supervisor_node(state: AgentState) -> Command:

    """This agent mainly serves as a router/decision maker on what the next visited node should be"""

    
    msgs = list(state.get("messages"))

    next = state.get("next")
    if next:
        return Command(goto=next, update={"messages": msgs, "next": None})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)
    decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke([SUPERVISOR_SYS, *msgs])

    if decision.reply:
        msg_reply = AIMessage(name="supervisor", content=decision.reply)
    else:
        msg_reply= AIMessage(name="supervisor", content=f"[route={decision.next}] {decision.reasoning}")

    update: AgentState = {
        "cur_reasoning": decision.reasoning,
        "messages": [*msgs, msg_reply],
        "next": decision.next,
    }

    if decision.receipt_uri:
        update["receipt_uri"] = decision.receipt_uri.strip()
    if decision.receipt_lines:
        update["receipt_lines"] = decision.receipt_lines.strip()
    if decision.query_text:
        update["query_text"] = decision.query_text.strip()
    if decision.rag_text:
        update["rag_text"] = decision.rag_text.strip()

    print(state.get("cur_reasoning"))


    return Command(goto=decision.next if decision.next != "FINISH" else END, update=update)

#remote adk_agent node

async def ocr_node(state: AgentState) -> Command:

    """Communicates with remote OCR agent using a2a protocol. The remote ADK agent extracts text from receipts"""

    msgs = list(state.get("messages"))
    uri = state.get("receipt_uri")

    print ("URI!!!!!!!!!!!!!!!", uri)
    if not uri:
        return Command(goto="supervisor", update={"errors": [*(state.get("errors") or []), "No receipt URI."], "messages": msgs})

    result = await ocr_client.send_ocr(uri)
    st = (result.get("state")).lower()

    if st == "completed":
        text = (result.get("text")).strip()
        if not text:
            return Command(goto="supervisor",
                           update={"errors": [(state.get("errors")), "OCR returned empty text."], "messages": msgs})
        return Command(
            goto="supervisor",
            update={
                "messages": [*msgs, AIMessage(name="ocr_node", content="OCR done.")], 
                "receipt_lines": text,
                "ocr_text": text,
                "receipt_uri": None,
                "next": "normalize_node",
            },
        )
    

    if st == "input_required":
        return Command(goto="supervisor",
                       update={
                           "messages": [*msgs, AIMessage(name="ocr_node", content=result.get("text"))],
                           "next": "FINISH",
                       })
    print("result", result.get('text'))
    return Command(goto="supervisor",
                   update={
                       "errors": [*(state.get("errors") or []), f"OCR error: {result.get('text')}"],
                       "messages": [*msgs, AIMessage(name="ocr_node", content=f"OCR error: {result.get('text')}")]
                   })

#normalize node

def normalize_node(state: AgentState) -> Command:

    """Takes The extracted Receipt Lines and extracts important info and normalizes them"""
    msgs = list(state.get("messages", []))
    lines = state.get("receipt_lines")

    print("LINEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEES", lines)

    if not lines:
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                lines = m.content.strip()
                break

    if not lines:
        return Command(goto="supervisor",
                       update={"errors": [*(state.get("errors") or []), "No lines to normalize."], "messages": msgs})

    agent = create_normalize_agent()
    resp = agent.invoke([HumanMessage(content=lines)]) 

    vendor = resp.vendor_name
    total = resp.total_amount

    normalized = resp.model_dump()

    msg = AIMessage(name="normalize_node", content=f"Normalized receipt from {vendor}, total {total}.")

    return Command(
        goto="supervisor",
        update={
            "messages": [*msgs, msg],
            "normalized_receipt": normalized,             
            "vendor_name": vendor,                        
            "receipt_category": resp.receipt_category,    
            "receipt_date": resp.date,                   
            "total_amount": total,                       
            "receipt_lines": None,
            "next": "store_node",
        },)

#store node

store_prompt = SystemMessage(
        content=(
            "You are a receipt‑storage assistant. "
            "When provided with a normalized receipt in JSON, you MUST call "
            "the `store_receipt` tool and pass the JSON as its argument. "
            "Do not answer with natural language; just call the tool."
        )
    )

async def store_node(state: AgentState) -> Command:

    """"Storage Agent that that uses provided tools to store normalized receipt into a database"""
    "entered store "


    msgs = list(state.get("messages", []))
    doc = state.get("normalized_receipt")

    print("doc!!!!!!!!!!!!!!!!!!!!!!!!! ", doc)

    if not doc:
        return Command(
            goto="supervisor",
            update={
                "errors": [*(state.get("errors") or []), "Nothing to store."],
                "messages": msgs,
            },
        )

    user_text = "Here is the receipt to store:"  
    conversation = msgs + [store_prompt, HumanMessage(content=json.dumps(doc))]

    async with stdio_client(SERVER_PARAMS_STORAGE) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await load_mcp_tools(session=session)
            tool_map = {t.name: t for t in tools}

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, api_key= GOOGLE_API_KEY)
            model = llm.bind_tools(list(tool_map.values()))

            ai_msg = await model.ainvoke(conversation)
            conversation.append(ai_msg)

            storage_results = []
            tool_errors = []

            if getattr(ai_msg, "tool_calls", None):
                outputs = []
                for tool_call in ai_msg.tool_calls:
                    try:
                        print(f"--- Invoking tool: {tool_call}")
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
                                tool_call_id=tool_call["id"],))

                conversation.extend(outputs)
                final_ai = await model.ainvoke(conversation)
                conversation.append(final_ai)
            else:
                final_ai = ai_msg
                tool_errors.append("Model did not call the storage tool.")


    status = "ok" if storage_results and not tool_errors else "failed"
    content = "Stored successfully" if status == "ok" else f"Storage failed: {tool_errors or 'unknown error'}"
    msg = AIMessage(name="store_node", content=content)

    return Command(
        goto="supervisor",
        update={
            "messages": [*msgs, msg],
            "storage_status": status,
            "storage_result": storage_results,
            "normalized_receipt": None,
            "next": "FINISH",
        },
    )

#query node

async def query_node(state: AgentState) -> Command:

    """Agent that retrieves information about user's receipt information from a database"""

    msgs = list(state.get("messages", []))
    q = state.get("query_text")
    if not q:
        return Command(goto="supervisor",
                       update={"errors": [*(state.get("errors") or []), "No query text."], "messages": msgs})

    payload = {"messages": [("user", q)]}
    async with stdio_client(SERVER_PARAMS_QUERY) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await load_mcp_tools(session=session)
            agent = create_query_agent(tools)
            res = await agent.ainvoke(payload)

    answer_text = getattr(res, "content", str(res))
    return Command(
        goto="supervisor",
        update={
            "messages": [*msgs, AIMessage(name="query_node", content=answer_text)],
            "query_result": {"content": answer_text},
            "query_text": None,
            "next": "FINISH",
        },
    )

#rag_node


tool_first_sys = SystemMessage(content=(
    "You are a RAG operator. You MUST call tools to answer. "
    "You have two options query from an sql database for specific prices in lebanon or articles_retrieval tool when asked about anything related to lebanese economy/inflation"
    "Never answer directly without calling at least one tool. "
    "Prefer SQL if the user asks for a price in a specific city. "
    "Workflow for SQL and prices in Lebanon always do this never skip a step: (1) call list_tables, (2) call tables_schema on tables, "
    "(3) call execute_sql to fetch the value, (4) answer concisely. The 4 are necessary"
    "Workflow for being asked about economy in Lebanon or inflation or something related: call articles_retrieval_tool"
    "Never ask user for table details"
    "When using execute tool, generate clear, accuracte SQL queries, only SELECT queries are allowed. Query on more than one table and compare results"
    "When querying for an item the item name may vary (e.g., 'Milk' vs 'Milk Powder (Full Fat)'): "
        "FIRST run: SELECT DISTINCT Item FROM product_prices_in_cities "
        "WHERE lower(Item) LIKE '%milk%'; "
        "Then choose the best match and run a precise SELECT."
    "If SQL querying isnt a success, try articles_retrieval_tool maybe you can find some info there about prices, let this be your last resort "
))

#reference : https://langchain-ai.github.io/langgraph/how-tos/tool-calling/?_gl=1*746vl1*_gcl_au*MTYwNDEzNjE0My4xNzU0NDk0Nzcx*_ga*NDQzOTExMDU2LjE3NTQ0OTQ3ODI.*_ga_47WX3HKKY2*czE3NTU3MjE2NDUkbzQ2JGcxJHQxNzU1NzIxNzgxJGo2MCRsMCRoMA..#toolnode

async def rag_node(state: AgentState) -> Command:
    """
    Execute a RAG query using LangGraph's ToolNode.
    """
    msgs: List[BaseMessage] = list(state.get("messages", []))
    q: Optional[str] = state.get("rag_text")
    if not q:
        return Command(
            goto="supervisor",
            update={
                "errors": [*(state.get("errors") or []), "No text."],
                "messages": msgs,
            },
        )

    async with stdio_client(SERVER_PARAMS_RAG) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await load_mcp_tools(session=session)

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key= GOOGLE_API_KEY)
            model = llm.bind_tools(list(tools))  
            tool_node = ToolNode(list(tools))    

            conversation: List[BaseMessage] = [tool_first_sys, HumanMessage(content=q)]

            response: AIMessage = await model.ainvoke(conversation)
            conversation.append(response)

            while getattr(response, "tool_calls", None):
                tool_state: Dict[str, Any] = await tool_node.ainvoke({"messages": [response]})
                tool_messages: List[ToolMessage] = tool_state.get("messages", [])
                conversation.extend(tool_messages)
                response = await model.ainvoke(conversation)
                conversation.append(response)

            answer_text: str = getattr(response, "content", str(response))

    return Command(
        goto="supervisor",
        update={
            "messages": [*msgs, AIMessage(name="rag_node", content=answer_text)],
            "rag_result": {"content": answer_text},
            "rag_text": None,
            "next": "FINISH",
        },
    )

ROUTES = {"ocr_node", "normalize_node", "store_node", "query_node", "rag_node"} 
def router(state: AgentState): 
    next = state.get("next") 
    if next == "FINISH": 
        return END 
    return next if next in ROUTES else END

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

async def run_single_graph_test(prompt: str) -> Dict[str, Any]:
    return await purr.ainvoke({"messages": [("user", prompt)]})

async def main():

    result = await run_single_graph_test("https://storage.googleapis.com/receipts_bucket_2023/receipts/1755821408.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=receipt-assitant%40aribnb-468618.iam.gserviceaccount.com%2F20250822%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250822T001047Z&X-Goog-Expires=900&X-Goog-SignedHeaders=host&response-content-disposition=inline&response-content-type=image%2Fjpeg&X-Goog-Signature=7ee60d570849f1ccc66ebd26262869443113a9c531a919db98f65f612e6153fad0fd17aa2b34f2130018bcc65e6045f418651ada5ac4897dc3bc9a0c2f280bec3d08b063e59f20d8be5b574b366d5771152a0be22b311a0508ba9bdb05403a73a562d01b9e329c871bb08dfd6cf81821c7734bcff53b1a79a90978cc23a9bdf045f86977d67903b62846b6177c3c86dc340aba591ca763a1ad3610b40bbbb63744e8e743d2ae0df066416f200c4795ec12de77cbfe47f6f88a12c11f5c7d688e6c177f2ed825b4be0231383adca3e7156369bd1f87fbf073ff7cac4e9467b848af7b1fd84f9fba3b5c138e1dc0dbbf22c54f03f1c44cc5d3a1ab530c4c04fb9c")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())