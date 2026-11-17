import uuid
import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph_database import chatbot, retrieve_all_threads

# ******************************** Page Config & Styling ********************************
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬", layout="wide")

st.markdown("""
<style>
    /* Sleek sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #121316;
    }

    /* "New Chat" primary button - prominent with vibrant accent */
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"],
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
        justify-content: center !important;
        text-align: center !important;
        margin-bottom: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover,
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.6) !important;
        transform: translateY(-1px);
    }

    /* Conversation thread cards - clear card background, left aligned */
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"],
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"],
    section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) {
        width: 100% !important;
        background-color: #1f232b !important;
        border: 1px solid #2e3440 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        margin-bottom: 6px !important;
        justify-content: flex-start !important;
        text-align: left !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"]:hover,
    section[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]):hover {
        background-color: #2b313d !important;
        border-color: #6366f1 !important;
        color: #ffffff !important;
        transform: translateX(2px);
    }

    /* Force text inside buttons to be left aligned with proper ellipsis */
    section[data-testid="stSidebar"] div.stButton button div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] div.stButton button p {
        text-align: left !important;
        font-size: 13.5px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Chat bubble spacing */
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 12px;
    }

    /* Center chat messages & constrain width like ChatGPT */
    .main .block-container {
        max-width: 820px !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 2rem !important;
        padding-bottom: 7rem !important;
        margin: 0 auto !important;
    }

    /* Center and constrain the chat input box at the bottom */
    div[data-testid="stBottom"] > div {
        max-width: 820px !important;
        margin: 0 auto !important;
    }

    div[data-testid="stChatInput"] {
        max-width: 820px !important;
        margin: 0 auto !important;
    }
</style>
""", unsafe_allow_html=True)


# ******************************** Utility functions ********************************
def generate_thread_id():
    return str(uuid.uuid4())


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
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_thread_title(thread_id):
    """Har thread ka pehla user question nikaal kar chhota title banata hai."""
    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = extract_text(msg.content).strip()
            if text:
                return text[:26] + ("..." if len(text) > 26 else "")
    return "New Chat"


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
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # Filter out any hardcoded legacy test thread
    raw_threads = retrieve_all_threads()
    st.session_state["chat_threads"] = [t for t in raw_threads if t != "thread-1"]


# ******************************** Sidebar ********************************
with st.sidebar:
    st.markdown("### 💬 LangGraph AI")

    if st.button("➕ New Chat", type="primary", use_container_width=True):
        reset_chat()
        st.rerun()

    st.markdown("---")
    st.caption("Recent Conversations")

    for thread_id in st.session_state["chat_threads"][::-1]:
        if thread_id == "thread-1":
            continue

        title = get_thread_title(thread_id)
        is_active = (thread_id == st.session_state["thread_id"])
        btn_label = f"📍 {title}" if is_active else f"💬 {title}"

        if st.button(btn_label, key=f"thread_{thread_id}", use_container_width=True):
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
            st.rerun()


# ******************************** Main chat UI ********************************
if not st.session_state["message_history"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>✨ How can I help you today?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Ask a question, analyze documents, or brainstorm ideas.</p>", unsafe_allow_html=True)
else:
    for message in st.session_state["message_history"]:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    # Thread sidebar me tabhi add karein jab user pehla message bheje
    add_thread(st.session_state["thread_id"])

    # Show and store the user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Stream and store the assistant reply
    with st.chat_message("assistant", avatar="🤖"):
        ai_message = st.write_stream(stream_ai_text(user_input))

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
    st.rerun()
