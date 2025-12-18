import requests
from dotenv import load_dotenv
# from langchain.agents import create_agent
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
load_dotenv()


@tool('get_weather', description='Get the weather in a given location', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'http://wttr.in/{city}?format=j1')
    return response.json()

@tool('get_time', description='Get the current time', return_direct=False)
def get_time():
    return {'time': requests.get('http://worldtimeapi.org/api/ip').json()['datetime']}
#
# agent = create_agent("claude-haiku-4-5-20251001", tools=[get_weather],
#                      system_prompt="You are a helpful weather assistant. Responde in funny manner")
llm = ChatOllama(
    # model="llama3.2:latest",  # lub inny model zainstalowany w Ollama
    model="qwen3:14b",  # lub inny model zainstalowany w Ollama
    base_url="http://localhost:11434",  # domyślny adres Ollama
    # base_url="https://ollama.slaskie.pl",  # domyślny adres Ollama
)

agent = create_agent(
    llm,
    tools=[get_weather, get_time],
    system_prompt="You are a helpful weather assistant. Respond  about  current  weather conditions from relevant tool in a funny manner."
)


response = agent.invoke({
    'messages': [
        {"role": "user", "content": "What is the weather like in London?"}
    ]
})

print(response['messages'][-1].content)
