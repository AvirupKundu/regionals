
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import httpx
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create OpenAI model client
client = OpenAIChatCompletionClient(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
    model=os.getenv("LLM_MODEL"),
    http_client=httpx.AsyncClient(verify=False)
)

# Define agents
writer = AssistantAgent(
    "writer",
    model_client=client,
    system_message="Draft a short paragraph on climate change with incorrect grammar."
)

editor1 = AssistantAgent(
    "editor1",
    model_client=client,
    system_message="Edit the paragraph for grammar."
)

editor2 = AssistantAgent(
    "editor2",
    model_client=client,
    system_message="Edit the paragraph for style."
)

final_reviewer = AssistantAgent(
    "final_reviewer",
    model_client=client,
    system_message="Choose the best version of the paragraph and explain why."
)

# Build workflow graph
builder = DiGraphBuilder()
builder.add_node(writer).add_node(editor1).add_node(editor2).add_node(final_reviewer)

builder.add_edge(writer, editor1)
builder.add_edge(writer, editor2)
builder.add_edge(editor1, final_reviewer)
builder.add_edge(editor2, final_reviewer)

graph = builder.build()

flow = GraphFlow(
    participants=builder.get_participants(),
    graph=graph,
)


# Run the flow and capture responses
async def main() -> None:
    # Run the flow
    result = await flow.run(task="Write a short paragraph about climate change.")

    # Extract responses
    responses = {
        "writer": "",
        "editor1": "",
        "editor2": "",
        "final_reviewer": ""
    }

    for msg in result.messages:
        if msg.source in responses:
            responses[msg.source] = msg.content

    # Print responses
    print("\nWriter Response:\n", responses["writer"])
    print("\nEditor1 Response:\n", responses["editor1"])
    print("\nEditor2 Response:\n", responses["editor2"])
    print("\nFinal Reviewer Response:\n", responses["final_reviewer"])

    # Save to JSON
    with open("agent_responses.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)

    print("\n✅ Responses saved to agent_responses.json")



# Entry point
if __name__ == "__main__":
    asyncio.run(main())
