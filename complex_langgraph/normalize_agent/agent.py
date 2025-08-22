import asyncio
import json
from typing import List, Literal, Optional
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY= "AIzaSyByctz3rxpT0W_s9bEV2lJUyXDDvXvP6Ys"

class Item(BaseModel):
    name: str
    type: Literal["item", "service", "fee", "tax", "other"]
    price: Optional[float]
    qty: Optional[float]

class ReceiptSchema(BaseModel):
    receipt_category: str
    vendor_name: str
    date: str
    total_amount: float
    items: List[Item]

system_prompt = SystemMessage(
    "You are a normalization agent.\n"
    "INPUT: translated receipt lines as plain text, one line per item, same order as the original.\n\n"
    "OUTPUT (STRICT JSON only; no code fences, no extra text):\n"
    "{\n"
    '  "receipt_category": "<concise category you infer>",\n'
    '  "vendor_name": "<best single-line vendor/issuer name>",\n'
    '  "date": "<best single-line date if available store it in month/day/year format, otherwise empty>",\n'
    '  "total_amount": <number>,\n'
    '  "items": [\n'
    '    {"name": "item name","type":"item|service|fee|tax|other","price": <number|null>, "qty": <number|null>}\n'
    '  ]\n'
    "}\n\n"
    "RULES:\n"
    "- Infer a SHORT category (e.g., 'internet', 'grocery', 'restaurant', ...).\n"
    "- Preserve numbers exactly; convert formatted numbers to numeric.\n"
    "- total_amount = final payable total.\n"
    "- vendor_name = issuer/merchant/org from header lines.\n"
    "- items: only include if reliable; use null for unknown price/qty; no extra fields.\n"
    "- Output must be valid JSON. No trailing commas, no comments, no markdown.\n"
    "if the receipt lines included a not null converetd ammount, use this as the total amount"
)

def create_normalize_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    llm_struct = llm.with_structured_output(ReceiptSchema)  
    return llm_struct


if __name__ == "__main__":
    agent = create_normalize_agent()
    query = "RAMMAL SUPERMARKET\n, 8/17/2025 \n,Cake: 25$\n, Another Cake 500$\n, total: 525$"
    result = agent.invoke(query)
    print(result.model_dump_json())


##with create_react_agent

# def create_normalize_agent():

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash")

#     agent = create_react_agent(model = llm, prompt= system_prompt, response_format= ReceiptSchema, tools =[])
#     return agent
