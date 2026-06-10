import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Gemini Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Gemini Chatbot")
st.caption("Built with LangChain + Google Gemini")

# --------------------------------------------------
# System Prompt
# --------------------------------------------------
SYSTEM_PROMPT = """
You are an intelligent AI chatbot designed to assist users.

Guidelines:
- Answer user questions clearly and accurately.
- Be helpful, professional, and concise.
- Retain context from previous messages.
- Explain technical concepts with examples when appropriate.
- If unsure about an answer, acknowledge uncertainty.
- Format responses neatly using Markdown.
"""

# --------------------------------------------------
# Initialize Chat Memory
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

# --------------------------------------------------
# Clear Chat Button
# --------------------------------------------------
col1, col2 = st.columns([1, 5])

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
        st.rerun()

# --------------------------------------------------
# Initialize Gemini Model
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for message in st.session_state.messages:

    # Skip system prompt from UI
    if isinstance(message, SystemMessage):
        continue

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --------------------------------------------------
# User Input
# --------------------------------------------------
user_query = st.chat_input("Ask me anything...")

if user_query:

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Store user message
    st.session_state.messages.append(
        HumanMessage(content=user_query)
    )

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = llm.invoke(
                st.session_state.messages
            )

            st.markdown(response.content)

    # Store AI response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )
