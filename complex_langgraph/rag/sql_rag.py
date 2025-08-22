import pandas as pd 
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Date, ForeignKey


engine = create_engine("sqlite:///C:/Users/saidy/OneDrive/Desktop/Smart_Receipt_Assistant/complex_langgraph/rag/lebanon_prices.db") 

df1 = pd.read_csv(r"C:/Users/saidy/Downloads/cost_of_living.csv").to_sql(name="cost_of_living", con=engine)
df2 = pd.read_csv(r"C:/Users/saidy/OneDrive/Desktop/PRODUCT_PRICES_APRIL_2025csv.csv").to_sql(name ="product_prices_in_cities", con = engine)




