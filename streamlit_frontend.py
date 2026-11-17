
import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Load conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # Display user message
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message('user'):
        st.write(user_input)

    # Get chatbot response
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]},
        config=CONFIG
    )

    # Get latest AI message
    ai_message = response['messages'][-1].content

    # Extract only text from response
    if isinstance(ai_message, list):
        ai_message = ''.join(
            block.get('text', '')
            for block in ai_message
            if isinstance(block, dict) and block.get('type') == 'text'
        )

    # Store only clean text
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message
    })

    # Display assistant response
    with st.chat_message('assistant'):
        st.write(ai_message)
