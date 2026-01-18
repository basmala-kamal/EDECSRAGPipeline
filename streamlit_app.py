import streamlit as st
import requests


st.title("Simple Chatbot Demo")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history

def display_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

def gen_response(user_message):
  url = "http://localhost:8000/documents/query"
  payload = {
    "question": user_message,
    # Optionally set document_id if needed
    # "document_id": "optional-uuid-to-filter"
  }
  try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json().get("answer", response.text)
  except Exception as e:
    return f"[API error: {e}]"

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    # Placeholder bot response
    bot_response = gen_response(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    display_messages()