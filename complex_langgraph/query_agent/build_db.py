from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date, ForeignKey


engine = create_engine("sqlite:///C:/Users/saidy/OneDrive/Desktop/hehe/receipts.db", echo=True)


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