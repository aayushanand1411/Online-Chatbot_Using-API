import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # stored in .env and file need to in same folder

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# System Prompt 
system_prompt = SystemMessage(
    content="""
You are a helpful AI chatbot designed to assist users with their questions.

Guidelines:
- Provide accurate and concise answers.
- Explain technical concepts with examples when appropriate.
- Maintain context from previous messages.
- If you do not know an answer, clearly state that.
- Be polite and professional.
"""
)

# Initialize history with system prompt
chat_history = [system_prompt]

print("Gemini Chatbot Started!")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    chat_history.append(
        HumanMessage(content=user_input)
    )

    response = llm.invoke(chat_history)

    chat_history.append(
        AIMessage(content=response.content)
    )

    print(f"\nBot: {response.content}\n")
