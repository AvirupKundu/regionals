from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import httpx
from dotenv import load_dotenv
 
# Load environment variables from .env
load_dotenv()
 
def llm_call(system_prompt: str, user_prompt: str) -> str:
    # Use consistent env keys
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    model = "gpt-4.1-nano"
 
    # In production, set verify=True for SSL
    client = httpx.Client(verify=False)
 
    # Initialize the OpenAI chat model
    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        http_client=client
    )
 
    # Define the system and user prompts
    system_msg = SystemMessage(content=system_prompt)
    user_msg = HumanMessage(content=user_prompt)
 
    # Call the LLM with both prompts
    try:
        response = llm.invoke([system_msg, user_msg])
        print(response.content)
        return response.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""
 
if __name__ == "__main__":
    system_prompt = "You are a helpful assistant."
    user_prompt = "Explain the theory of relativity in 10 words only."
    llm_call(system_prompt, user_prompt)