# Custom AI Agent Chatbot

This repository contains a Streamlit-based AI chatbot that integrates with Groq and OpenAI via LangChain and offers optional web search capabilities via Tavily.

## Features
- Toggle between Groq (`llama-3.3-70b-versatile`) and OpenAI (`gpt-4o-mini`)
- Role-based system prompt customization
- Optional Tavily-powered web search
- Lightweight single-file (`app.py`) architecture for easy deployment

## Usage

```bash
conda activate custom_chatbot_env
pip install -r requirements.txt
streamlit run app.py
