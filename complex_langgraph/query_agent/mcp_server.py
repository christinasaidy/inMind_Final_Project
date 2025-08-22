
from __future__ import annotations
from langchain_community.utilities import SQLDatabase
import os, time, requests, mimetypes
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import pandas as pd
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDataBaseTool,
)
from datetime import datetime
from sqlalchemy import create_engine

load_dotenv()

#refernce : "https://medium.com/@ayush4002gupta/building-an-llm-agent-to-directly-interact-with-a-database-0c0dd96b8196"


mcp = FastMCP("query-agent")

db = SQLDatabase.from_uri("sqlite:///C:/Users/saidy/OneDrive/Desktop/Smart_Receipt_Assistant/receipts.db")  

@mcp.tool()
def get_current_date() -> str:
    """Get the current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


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

     mcp.run(transport='stdio')




