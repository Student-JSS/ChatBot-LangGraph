import uuid

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from langgraph_database import chatbot, retrieve_all_threads


# ******************************** Utility functions ********************************
def generate_thread_id():
    return uuid.uuid4()


def extract_text(content):
    """
    Newer models (e.g. Gemini with thinking) return content as a list of blocks:
    [{"type": "text", "text": "...", "index": 0}, {"extras": {"signature": ...}}]
    This returns only the plain text, ignoring thinking/signature blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def stream_ai_text(user_input):
    """Yield only the text of AI chunks so st.write_stream gets plain strings."""
    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": st.session_state["thread_id"]}},
        stream_mode="messages",
    ):
        if isinstance(message_chunk, AIMessageChunk):
            text = extract_text(message_chunk.content)
            if text:
                yield text


# ******************************** Session state ********************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# ******************************** Sidebar ********************************
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue  # skip tool/system messages

            text = extract_text(msg.content)
            if text:
                temp_messages.append({"role": role, "content": text})

        st.session_state["message_history"] = temp_messages


# ******************************** Main chat UI ********************************
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show and store the user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Stream and store the assistant reply
    with st.chat_message("assistant"):
        ai_message = st.write_stream(stream_ai_text(user_input))

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )