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
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    else:
        return "❌ Invalid provider"

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    # state_modifier argument only works in updated langgraph version
    # Ensure your Streamlit Cloud uses compatible langgraph-prebuilt==0.6.7
    agent = create_react_agent(
        model=llm,
        tools=tools,
        #system_prompt=system_prompt  # Keep this line with compatible version
    )

    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
    return ai_messages[-1] if ai_messages else "⚠️ No response from AI."

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Custom AI Chatbot", layout="centered")
st.title("🤖 Custom AI Agent Chatbot")
st.write("Interact with your custom AI Agent using Groq / OpenAI + Web Search!")

# System Prompt
system_prompt = st.text_area(
    "🔧 Define your AI Agent role:",
    height=70,
    placeholder="For example: Civil Engineer, Data Analyst, Teacher..."
)

# Provider & Model Selection
provider = st.radio("Select Provider:", ("Groq", "OpenAI"))

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
else:
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

# Search Option
allow_web_search = st.checkbox("Allow Web Search")

# User Query
user_query = st.text_area(
    "💬 Enter your query:",
    height=150,
    placeholder="Ask anything from your AI Agent..."
)

# Buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Ask Agent!"):
        if user_query.strip():
            with st.spinner("Thinking..."):
                response = get_response_from_ai_agent(
                    llm_id=selected_model,
                    query=[user_query],
                    allow_search=allow_web_search,
                    system_prompt=system_prompt,
                    provider=provider
                )
            st.subheader("Agent Response")
            st.markdown(f"**{response}**")

with col2:
    if st.button("🔄 Reset"):
        st.session_state["system_prompt"] = ""
        st.session_state["user_query"] = ""
        st.session_state["selected_model"] = "llama-3.3-70b-versatile"
        st.session_state["provider"] = "Groq"
        st.session_state["allow_web_search"] = False
        st.experimental_rerun()
