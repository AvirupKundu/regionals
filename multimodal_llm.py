from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import httpx
from dotenv import load_dotenv
import base64
import imghdr

load_dotenv()

def _image_to_data_url(path: str) -> str:
    """Read a local image file and return a data URL (image/*;base64,...)"""
    with open(path, "rb") as f:
        raw = f.read()
    kind = imghdr.what(None, raw) or "png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{kind};base64,{b64}"

def llm_call(system_prompt: str, user_prompt: str | None = None, image_path: str | None = None):
    """
    Send a chat request to the model. Supports:
      - text only: provide user_prompt, leave image_path None
      - image only: provide image_path, leave user_prompt None
      - image + text: provide both

    Images are passed as structured multimodal inputs, not inline Markdown.
    """
    base_url = os.getenv("api_endpoint")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL")

    client = httpx.Client(verify=False)

    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        http_client=client
    )

    system_message = SystemMessage(content=system_prompt)

    # Build multimodal content
    parts = []
    if user_prompt:
        parts.append({"type": "text", "text": user_prompt})

    if image_path:
        try:
            data_url = _image_to_data_url(image_path)
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read/encode image: {e}")

    if not parts:
        raise ValueError("Provide at least user_prompt or image_path")

    user_message = HumanMessage(content=parts)

    # Invoke the model
    response = llm.invoke([system_message, user_message])

    print(response.content)
    return response.content

if __name__ == "__main__":
    # Example 1: Text only
    # llm_call("You are a helpful assistant.", user_prompt="Explain special relativity simply.")

    # Example 2: Image only
    # llm_call("You are a helpful assistant.", image_path="Designer.png")

    # Example 3: Image + text
    llm_call(
        "You are a helpful assistant.",
        user_prompt="Summarize what is shown and list any visible errors.",
        image_path="Designer.png"
    )
