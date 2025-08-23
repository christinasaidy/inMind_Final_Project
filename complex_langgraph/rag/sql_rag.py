import pandas as pd 
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date, ForeignKey

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_DIR = PROJECT_ROOT / "databases"

DB_PATH = DB_DIR / "lebanon_prices.db"

engine = create_engine(f"sqlite:///{DB_PATH}")


df1 = pd.read_csv(r"C:/Users/saidy/Downloads/cost_of_living.csv").to_sql(name="cost_of_living", con=engine)
df2 = pd.read_csv(r"C:/Users/saidy/OneDrive/Desktop/PRODUCT_PRICES_APRIL_2025csv.csv").to_sql(name ="product_prices_in_cities", con = engine)




