import os
import time
from datetime import timedelta
from typing import List, Tuple

import requests
import gradio as gr
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

def generate_signed_url(image_path: str) -> str:
    client = storage.Client()
    blob_name = f"receipts/{int(time.time())}.jpg"
    bucket = client.bucket("receipts_bucket_2023")
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(image_path, content_type="image/jpeg")
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=15),
        method="GET",
        response_disposition="inline",
        response_type="image/jpeg",
    )
    return url

def call_backend(messages: List[str], backend_url: str = "http://localhost:8001/chat") -> str:
    payload = {"messages": messages}
    response = requests.post(backend_url, json=payload)
    response.raise_for_status()
    return response.text

def create_interface() -> gr.Blocks:
    """GRADIO UI """


    CSS = """
    html, body, .gradio-container { height: 100%; }
    .gradio-container {
        min-height: 100vh;
        display: grid;
        place-items: center;       /* vertical + horizontal centering base */
        padding: 24px;
    }

    /* FORCE horizontal centering regardless of Gradio wrapper widths */
    #app {
        width: clamp(360px, 80vw, 900px);
        position: relative;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    /* Center the title and control rows */
    #title { text-align: center; }
    #textrow, #controls { justify-content: center; }

    /* Make the chatbot span the app width */
    #chatbot { width: 100%; }
    """

    with gr.Blocks(title="LangGraph Chatbot with Image Upload", css=CSS, fill_height=True) as demo:
        state = gr.State([])

        with gr.Column(elem_id="app"):
            gr.Markdown("# Receipt Assistant", elem_id="title")

            chatbot = gr.Chatbot(height=520, elem_id="chatbot")

            with gr.Row(elem_id="textrow"):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Type a message and press Enter...",
                    scale=12,
                )

            with gr.Row(elem_id="controls"):
                upload_btn = gr.UploadButton(
                    "Upload image",
                    file_types=["image"],
                    file_count="single",
                    type="filepath",
                    size="md",
                    variant="secondary",
                )
                send_btn = gr.Button("Send", variant="primary")

            def add_message(history: List[Tuple[str, str]], role: str, content: str):
                history = history.copy()
                history.append((role, content))
                return history

            def handle_user_input(user_text: str, history: List[Tuple[str, str]]):
                if not user_text:
                    return history, history
                history = add_message(history, "user", user_text)
                messages = [msg for _, msg in history]
                try:
                    assistant_reply = call_backend(messages)
                except Exception as exc:
                    assistant_reply = f"Error communicating with backend: {exc}"
                history = add_message(history, "assistant", assistant_reply)
                return history, history

            def handle_image_upload(file_path: str, history: List[Tuple[str, str]]):
                if not file_path:
                    return history, history
                try:
                    signed_url = generate_signed_url(file_path)
                    
                except Exception as exc:
                    signed_url = f"Error uploading image: {exc}"
                history = add_message(history, "user", signed_url)
                messages = [msg for _, msg in history]
                try:
                    assistant_reply = call_backend(messages)
                except Exception as exc:
                    assistant_reply = f"Error communicating with backend: {exc}"
                history = add_message(history, "assistant", assistant_reply)
                return history, history

            send_btn.click(handle_user_input, inputs=[txt, state], outputs=[chatbot, state])
            txt.submit(handle_user_input, inputs=[txt, state], outputs=[chatbot, state])
            upload_btn.upload(handle_image_upload, inputs=[upload_btn, state], outputs=[chatbot, state])

    return demo

def main() -> None:
    demo = create_interface()
    demo.launch()

if __name__ == "__main__":
    main()