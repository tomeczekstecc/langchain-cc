from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@dataclass
class Context:
    user_id: str


@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


@tool("get_weather", description="Get the weather in a given location", return_direct=False)
def get_weather(city: str):
    try:
        response = requests.get(f"http://wttr.in/{city}?format=j1", timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to fetch weather data: {e}"}


@tool("get_time", description="Get the current time", return_direct=False)
def get_time():
    try:
        response = requests.get("http://worldtimeapi.org/api/ip", timeout=20)
        response.raise_for_status()
        return {"time": response.json()["datetime"]}
    except requests.RequestException as e:
        return {"error": f"Failed to fetch time data: {e}"}


# IMPORTANT: return_direct=False so the agent can still produce a final structured response
@tool("locate_user", description="Locate the user based on context", return_direct=False)
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "123":
            return "Vienna"
        case "456":
            return "Warsaw"
        case _:
            return "London"


llm = ChatOllama(
    # model="llama3.2:latest",
    model="qwen3:4b",
    base_url="http://localhost:11434",
)

model = init_chat_model("claude-haiku-4-5-20251001")
checkpointer = InMemorySaver()

agent = create_agent(
    # model=model,  # use the model instance you configured
    model=llm,  # use the model instance you configured
    tools=[get_weather, get_time, locate_user],
    system_prompt=(
        "You are a helpful weather assistant. "
        "Respond about current weather conditions from the relevant tool in a funny manner. "
        "Always produce the final answer in the required structured format."
    ),
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": 1}}
context = Context(user_id="456")


def print_agent_result(result: dict):
    """
    Prefer structured_response if present; otherwise print the last assistant message
    so the program doesn't crash and you can see what came back.
    """
    structured = result.get("structured_response")
    if structured is not None:
        print(structured)
        return

    messages = result.get("messages") or []
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None) or getattr(last, "text", None) or str(last)
        print(content)
        return

    # Last resort: print the whole dict (useful for debugging response shape)
    print(result)


response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather like?"}
        ],
    },
    config=config,
    context=context,
)

print_agent_result(response)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Is this usual at this time of the year?"}
        ],
    },
    config=config,
    context=context,
)

print_agent_result(response)