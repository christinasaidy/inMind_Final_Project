from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date, ForeignKey
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "databases"
DB_PATH = DB_DIR / "receipts.db"

engine = create_engine(f"sqlite:///{DB_PATH}")


metadata = MetaData()

receipts = Table(
    "receipts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("receipt_category", String, nullable=False),
    Column("vendor_name", String, nullable=False),
    Column("date", String, nullable=True),
    Column("total_amount", Float, nullable=False)
)

receipt_items = Table(
    "receipt_items", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("receipt_id", Integer, ForeignKey("receipts.id"), nullable=False),
    Column("name", String, nullable=True), 
    Column("type", String, nullable=False), 
    Column("price", Float, nullable=True),
    Column("qty", Float, nullable=True)
)

metadata.create_all(engine)

#reference : "https://sibabalwesinyaniso.medium.com/connecting-to-a-database-and-creating-tables-using-sqlalchemy-core-52cb79e51ca4"