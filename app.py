import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# ----------------- Backend Logic -----------------
def get_response_from_ai_agent(llm_id, query, allow_search, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    else:
        return "❌ Invalid provider"

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    # system_prompt not supported in langgraph-prebuilt==0.6.4
    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages", [])
    ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
    return ai_messages[-1] if ai_messages else "⚠️ No response from AI."

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Custom AI Chatbot", layout="centered")
st.title("🤖 Custom AI Agent Chatbot")
st.write("Interact with your custom AI Agent using Groq / OpenAI + Web Search!")

# Initialize session_state
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = ""
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "llama-3.3-70b-versatile"
if "provider" not in st.session_state:
    st.session_state["provider"] = "Groq"
if "allow_web_search" not in st.session_state:
    st.session_state["allow_web_search"] = False

# System Prompt
st.session_state["system_prompt"] = st.text_area(
    "🔧 Define your AI Agent role:",
    value=st.session_state["system_prompt"],
    height=70,
    placeholder="For example: Civil Engineer, Data Analyst, Teacher..."
)

# Provider & Model Selection
st.session_state["provider"] = st.radio(
    "Select Provider:",
    ("Groq", "OpenAI"),
    index=0 if st.session_state["provider"]=="Groq" else 1
)

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

if st.session_state["provider"] == "Groq":
    st.session_state["selected_model"] = st.selectbox(
        "Select Groq Model:",
        MODEL_NAMES_GROQ,
        index=MODEL_NAMES_GROQ.index(st.session_state["selected_model"]) 
              if st.session_state["selected_model"] in MODEL_NAMES_GROQ else 0
    )
else:
    st.session_state["selected_model"] = st.selectbox(
        "Select OpenAI Model:",
        MODEL_NAMES_OPENAI,
        index=MODEL_NAMES_OPENAI.index(st.session_state["selected_model"]) 
              if st.session_state["selected_model"] in MODEL_NAMES_OPENAI else 0
    )

# Search Option
st.session_state["allow_web_search"] = st.checkbox(
    "Allow Web Search",
    value=st.session_state["allow_web_search"]
)

# User Query
st.session_state["user_query"] = st.text_area(
    "💬 Enter your query:",
    value=st.session_state["user_query"],
    height=150,
    placeholder="Ask anything from your AI Agent..."
)

# Buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Ask Agent!"):
        if st.session_state["user_query"].strip():
            with st.spinner("Thinking..."):
                response = get_response_from_ai_agent(
                    llm_id=st.session_state["selected_model"],
                    query=[st.session_state["user_query"]],
                    allow_search=st.session_state["allow_web_search"],
                    provider=st.session_state["provider"]
                )
            st.subheader("Agent Response")
            st.markdown(f"**{response}**")

with col2:
    if st.button("🔄 Reset"):
        # Reset all session_state variables
        st.session_state["system_prompt"] = ""
        st.session_state["user_query"] = ""
        st.session_state["selected_model"] = "llama-3.3-70b-versatile"
        st.session_state["provider"] = "Groq"
        st.session_state["allow_web_search"] = False
        # Streamlit 1.49.1 compatible: no rerun
        st.experimental_set_query_params()
