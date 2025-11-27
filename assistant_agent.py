
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Define a model client
model_client = OpenAIChatCompletionClient(
    base_url = os.getenv("BASE_URL"),
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
    model= os.getenv("LLM_MODEL"),
    http_client = httpx.AsyncClient(verify=False)
)

# Define a simple function tool
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."

# Define an AssistantAgent
agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# Run the agent and stream messages
async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York?"))
    await model_client.close()

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
