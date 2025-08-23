from typing import Dict, Any, List
from langchain_core.tools import create_retriever_tool
from langchain_core.tools import create_retriever_tool
import os
from dotenv import load_dotenv
from articles_rag import article_vectorstore
from langchain_community.utilities import SQLDatabase
from mcp.server.fastmcp import FastMCP
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDataBaseTool,
)

from pathlib import Path


os.environ.setdefault("USER_AGENT", "christina-mcp/0.1")


load_dotenv()

mcp = FastMCP("rag-server", host="127.0.0.1", port=9001)



#load db
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "databases"
DB_PATH = DB_DIR / "lebanon_prices.db"

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")


@mcp.tool()
def articles_retrieval_tool(query: str) -> str:
    """Retrieve the most relevant documents related to economy, inflation, news in Lebanon"""
    retriever = article_vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(retriever,"retrieve_budget_info", query)
    results = retriever_tool.run(query)
    return results

@mcp.tool()
def list_tables() -> str:
    """List the available tables in the database"""
    return ListSQLDatabaseTool(db=db).invoke("")

@mcp.tool()
def tables_schema(tables: str) -> str:
    """
    Input is a comma-separated list of tables, output is the schema and sample rows
    for those tables. Be sure that the tables actually exist by calling `list_tables` first!
    Example Input: table1, table2, table3
    """
    tool = InfoSQLDatabaseTool(db=db)
    return tool.invoke(tables)

@mcp.tool()
def execute_sql(sql_query: str) -> str:
    """Execute a SQL query against the database. Returns the result"""
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
