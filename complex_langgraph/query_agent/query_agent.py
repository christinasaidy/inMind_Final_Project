import logging
import os
import asyncio
import json
import httpx

from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
load_dotenv()

GOOGLE_API_KEY= "AIzaSyByctz3rxpT0W_s9bEV2lJUyXDDvXvP6Ys"

system_message =  SystemMessage("""You are a SQL expert with strong attention to detail.

    You have FOUR tools:
    - list_tables(): lists all tables
    - tables_schema(tables: str): takes a comma-separated list of table names and returns their schema and sample rows
    - execute_sql(sql_query: str): executes a single SELECT query and returns results
    - get_current_date(): returns the current date in YYYY-MM-DD format just so youre always in the loop of what today is 

    MANDATORY WORKFLOW:
    1) ALWAYS call list_tables() first to see available tables.
    2) THEN call tables_schema() ONLY for tables that seem relevant to the question.
    3) Based on the schema, construct ONE safe SELECT query (no DML/DDL). Never select *; only needed columns.
    4) Call execute_sql() with that query.
    5) If execute_sql fails or returns empty, refine the query and try again (loop at most 3 times).
    6) Then answer the user in plain text.

    RULES:
    - Do not ask the user for table or column names unless you have already called list_tables() and tables_schema() 
    - Always query both tables for optimal results
    - LIMIT results to at most 5 rows unless the user requested another number or show all.
    - Never perform INSERT/UPDATE/DELETE/DROP/ALTER/etc.
    - "When querying for an item or a specific vendor or a specific category the name may vary: "
        "FIRST run: SELECT DISTINCT Item FROM receipts or receipt_items "
        "WHERE lower(Item) LIKE '%milk%'; "
        "Then choose the best match and run a precise SELECT."
                                
    """)


def create_query_agent(tools):
#   """Initializes a LangGraph agent that queries from database given a prompt"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        max_output_tokens=1024,
        top_p=1.0,
        top_k=40,
        api_key=GOOGLE_API_KEY
    )
    agent = create_react_agent(model = llm, tools = tools, prompt = system_message)
    return agent

##test 

# async def main():
#     agent = await create_query_agent()

#     result = await agent.ainvoke("What did i spend on clothes?")
#     print(result.content)  

# if __name__ == "__main__":
#     asyncio.run(main())
