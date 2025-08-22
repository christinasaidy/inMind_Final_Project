import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import insert
from sqlalchemy.engine import Engine
from sqlalchemy import MetaData, Table
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, List
from sqlalchemy import inspect



mcp = FastMCP("storage-agent")

db = create_engine("sqlite:///C:/Users/saidy/OneDrive/Desktop/Smart_Receipt_Assistant/receipts.db")  
# inspector = inspect(db) 
# print("HELOOOOOOOOOOOOOOOO")
# print(inspector.get_table_names())


metadata = MetaData()
metadata.reflect(bind=db) 
receipts = metadata.tables["receipts"]
receipt_items = metadata.tables["receipt_items"]


@mcp.tool()
def insert_receipt(data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert receipt and item into db."""
    with db.connect() as connection:
        r = connection.execute(insert(receipts).values(
            receipt_category=data["receipt_category"],
            vendor_name=data["vendor_name"],
            date=data.get("date"),
             total_amount=(
                    data.get("converted_total_amount")
                    or data.get("total_converted")
                    or data.get("total_amount")
                ),
        ))
        receipt_id = r.inserted_primary_key[0]
        
        items: List[Dict[str, Any]] = data.get("items", [])
        if items:
            connection.execute(
                insert(receipt_items),
                [
                    {
                        "receipt_id": receipt_id,
                        "name": it.get("name"),
                        "type": it["type"],
                        "price": it.get("price"),
                        "qty": it.get("qty"),
                    }
                    for it in items
                ],
            )
        connection.commit()
    return {"status": "ok", "receipt_id": receipt_id, "items_inserted": len(items)}

if __name__ == "__main__":
    mcp.run(transport="stdio")

#references : 
# https://sibabalwesinyaniso.medium.com/inserting-data-into-sqlite-using-sqlalchemy-core-9132772154e3
# "https://sibabalwesinyaniso.medium.com/updating-data-in-sqlite-using-sqlalchemy-core-5a5b75f5d4e7"
# https://github.com/applied-gen-ai/txt2sql/blob/main/create_tables.ipynb