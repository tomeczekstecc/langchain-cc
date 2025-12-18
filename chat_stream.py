import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
load_dotenv()

llm = ChatOllama(
    # base_url="https://ollama.slaskie.pl",
    base_url="http://localhost:11434",
    model="qwen3:4b"
)

# model = init_chat_model(model="claude-haiku-4-5-20251001")
# model = init_chat_model(model="claude-sonnet-4-5-20250929")

model = llm



consversation = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is Python programming language?"),
    AIMessage(content="Python is a programming language."),
    HumanMessage(content="What is programming language?")
]

for chunck in model.stream(consversation):
    print(chunck.text, end="", flush=True)