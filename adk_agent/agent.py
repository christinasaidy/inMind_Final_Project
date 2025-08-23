import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams, StdioServerParameters)
from dotenv import load_dotenv
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.cloud import storage
from datetime import timedelta
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams

load_dotenv()
from google.genai import Client
import os

VERTEXAI_PROJECT="aribnb-468618"
VERTEXAI_LOCATION="us-central1"

# translate_sub_agent = LlmAgent(
#         model="gemini-2.5-flash",
#         name = "Translate_OCR_agent",
#         description = "An agent to translate text extracted from receipt images using Azure Computer Vision Read 3.2.",
#         instruction = "You will translate the text extracted from receipt images to the english if its not in english.",)



def create_ocr_agent() -> LlmAgent:
    return LlmAgent(
        model="gemini-2.5-flash",
        name="Receipt_OCR_agent",
        description="Extracts text then converts totals to USD.",
        instruction=(
            "You have two tools: `ocr_read` and `convert_currency`.\n, always use them"
            "1) Always call `ocr_read` first (use image_url or [HOST_IMAGE_B64] when present).\n"
            "2) From the OCR result, detect the currency (e.g., symbol €, AED, LBP(or Lebanese Lira might be expressed usually LL).\n"
            "   If not obvious, default EUR when a € symbol appears, and in arabic default to lebanese liras LBP OR LL.\n"
            "3) Extract the receipt TOTAL amount (single number) and the receipt date if present "
            "(format YYYY-MM-DD; otherwise use 'latest').\n"
            "4) Call `convert_currency` with amount=<TOTAL>, from=<detected>, to='USD', currency_date=<date or 'latest'>.\n"
            "5) Return one final JSON object only with full {ocr text} that is transalted and has its currency replaced:\n"
            "If OCR fails, return the ocr_read JSON and stop."
            "translate the receipt text to english if its not in english."
            "6) uris from google cloud that are long, you will not touch them normalize them or east them examples: https://storage.googleapis.com/receipts_bucket_2023/receipts/1755769423.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=receipt-assitant%40aribnb-468618.iam.gserviceaccount.com%2F20250821%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250821T094400Z&X-Goog-Expires=900&X-Goog-SignedHeaders=host&response-content-disposition=inline&response-content-type=image%2Fjpeg&X-Goog-Signature=0a9e27110466bc7ded740acfede1700937a11fc83a525c34b01fd9b94cd1a9b7fecdab6426f466aa5fd0cab224dc20919d658e2eb2b81f13a07057e34d3e0e43e321c9b4a555684fe4fd7f6af0011a54a96e97eca91849b8f4889c6375c5a729470c0256019adfeebb0f0c446131ad46dad6abfc7cdb3046f63db049d7c2c4aabfe8fbdabb0df926b0fd0087035f343939d86dd260925f10f0ca245e0890d2280a720443d08fcf1ef2345946dffd88c45eaeb99107fe8e86d2291e84d1c4ef761269e8bba33adbb70c7a708bca6512700a08fcbb9caccba6e423c880fd4fa2ba933a9dc7429ebfb6d0b8d1d0ab0c942ff0aacdadf137a10390a507ceaf36ad9e"
        ),
        tools=[
            MCPToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url="http://127.0.0.1:9003/mcp/"  
                )
            )
        ],
        # sub_agents=[translate_sub_agent],
    )


#####test
APP_NAME   = "receipt_app"
USER_ID    = "tester"
SESSION_ID = "test_session"

IMAGE_URL = "https://storage.googleapis.com/receipts_bucket_2023/receipts/1755819491.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=receipt-assitant%40aribnb-468618.iam.gserviceaccount.com%2F20250821%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250821T233848Z&X-Goog-Expires=900&X-Goog-SignedHeaders=host&response-content-disposition=inline&response-content-type=image%2Fjpeg&X-Goog-Signature=1b315877650e30821ae5f73d833c4fd03816d9aad188d4920bcd55f4384c2488439028569f9d6a4c952f0cf9fb4cdfad8c30fcf76c222747109daf336401f23793b62d69a5571bbc280ef3ee1c2ccae4a5caa67b191bce1a5803f8b6dd2fc39d55876930ad0b2c14b10b6278d4550ac7c3319bca0368ff72bf856455a024555dc39a66e73aa225ca65e35b9d3c6fe694dc0ff267d8c6772b07d899ece580d33682b93f531e2149cbc4fd8c6b15fa97dc75cab7be33193d54a6edde3715c791ec9ec8b754020f350f3a00ccc3741befb71f61c2e7851c07aab5adf34c14df323a5afaa26a3e3be825416e324e3ced34f18cc3ee3be5076ba5bc8eb16c79f8ced8"

async def main():

    agent = create_ocr_agent()

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    prompt = (
        f"{IMAGE_URL}"
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_text = None
    for event in runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response():
            final_text = event.content.parts[0].text

    print("Final response:", final_text)


if __name__ == "__main__":
    asyncio.run(main())