import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(model="claude-haiku-4-5-20251001")

consversation = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is Python programming language?"),
    AIMessage(content="Python is a programming language."),
    HumanMessage(content="What is programming language?")
]

response = model.invoke(consversation)

print(response.content)
