from __future__ import annotations
import os, time, requests, mimetypes
from mcp.server.fastmcp import FastMCP
import httpx
from dotenv import load_dotenv
import os, httpx
from typing import Optional, Dict, Any

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
READ_URL = AZURE_ENDPOINT + "vision/v3.2/read/analyze"

if not AZURE_KEY or not AZURE_ENDPOINT:
    raise RuntimeError(".env variables AZURE_KEY and AZURE_ENDPOINT must be set correctly.")


mcp = FastMCP("receipt-ocr", host="127.0.0.1", port=9003)

@mcp.tool()
def get_current_date() -> str:
    """Get the current date in YYYY-MM-DD format."""
    return time.strftime("%Y-%m-%d")

BASE = "https://api.exchangerate.host"
@mcp.tool()

def convert_currency(
    amount: float,
    currency_from: str,
    currency_to: str = "USD",
    date: Optional[str] = None,       
    timeout_seconds: float = 10.0,
) -> Dict[str, Any]:
    """
    Coverts currencies 
    Returns: {"status": "ok"|"error", "converted": float|None, "rate": float|None, "date": str|None, "raw": dict}
    """
    params = {
        "from": currency_from.upper(),
        "to": currency_to.upper(),
        "amount": float(amount),
    }
    if date:                     
        params["date"] = date

    access_key = os.getenv("EXCHANGERATE_HOST_KEY")
    if access_key:
        params["access_key"] = access_key

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            r = client.get(f"{BASE}/convert", params=params)
            r.raise_for_status()
            data = r.json()

            success = data.get("success", True)
            if not success:
                return {"status": "error", "converted": None, "rate": None, "date": data.get("date"), "raw": data}

            result = data.get("result")
            info = data.get("info") or {}
            rate = info.get("quote", info.get("rate"))
            if rate is None and result is not None and amount != 0:
                rate = float(result) / float(amount)

            return {"status": "ok", "converted": result, "rate": rate, "date": data.get("date"), "raw": data}

    except httpx.HTTPStatusError as e:
        return {"status": "error", "converted": None, "rate": None, "date": None, "raw": {"error": str(e)}}
    except Exception as e:
        return {"status": "error", "converted": None, "rate": None, "date": None, "raw": {"error": str(e)}}
    

@mcp.tool(
    name="ocr_read",
    description="Extract text from a receipt image using Azure Computer Vision Read 3.2"
)
def ocr_read(image_url: str = None, image_path: str = None, timeout_seconds: int = 60):
    if not image_url and not image_path:
        return {"status": "failed", "error": "Provide image_url or image_path."}

    print("image urlllllllllllllllllllllllllllllllllllllllll", image_url)
    if image_url:
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_KEY,
            "Content-Type": "application/json",
        }
        resp = requests.post(READ_URL, headers=headers, json={"url": image_url})
    else:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_KEY,
            "Content-Type": "application/octet-stream",
        }
        resp = requests.post(READ_URL, headers=headers, data=image_bytes)

    resp.raise_for_status()
    op_url = resp.headers.get("Operation-Location")
    if not op_url:
        return {"status": "failed", "error": "No Operation-Location returned"}

    headers = {"Ocp-Apim-Subscription-Key": AZURE_KEY}
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result_resp = requests.get(op_url, headers=headers)
        result_resp.raise_for_status()
        result_data = result_resp.json()
        if result_data.get("status") in ["succeeded", "failed"]:
            break
        time.sleep(1)

    if result_data.get("status") != "succeeded":
        return {"status": result_data.get("status"), "raw": result_data}


    lines = []
    full_text_parts = []
    for page in result_data.get("analyzeResult", {}).get("readResults", []):
        for line in page.get("lines", []):
            lines.append(line.get("text", ""))
            full_text_parts.append(line.get("text", ""))

    return {
        "status": "succeeded",
        "full_text": "\n".join(full_text_parts),
        "lines": lines,
        "raw": result_data
    }



if __name__ == "__main__":
    mcp.run(transport="streamable-http")


