import streamlit as st
from agents.controller_agent import Controller
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border: 1px solid #333;
    }
    .stTextInput > div > div > input::placeholder {
        color: #888;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #1a1a1a;
        border-left: 4px solid #2196F3;
    }
    .chat-message.assistant {
        background-color: #0d0d0d;
        border-left: 4px solid #4CAF50;
    }
    .chat-message .message-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }
    .chat-message.user .message-header {
        color: #64B5F6;
    }
    .chat-message.assistant .message-header {
        color: #81C784;
    }
    .chat-message .message-content {
        color: #e0e0e0;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("AI Knowledge Assistant")
st.markdown("Ask questions and get intelligent responses")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize controller (cached to avoid recreation)
if "controller" not in st.session_state:
    st.session_state.controller = Controller(
        IntentAgent(model_path="models/intent_model.pkl"),
        RetrievalAgent(),
        VisionAgent(),
        ReasoningAgent(api_key=os.getenv("GROQ_API_KEY"))
    )

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user">
                    <div class="message-header">YOU</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="message-header">ASSISTANT</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

# Input area at the bottom
st.markdown("---")

# Initialize input counter if not exists
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Your question:",
        key=f"user_input_{st.session_state.input_counter}",
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

# Handle user input
if send_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show thinking indicator
    with st.spinner("Thinking..."):
        # Get response from controller
        response = st.session_state.controller.handle_query(user_input)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear the input by using a counter to reset the key
    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0
    st.session_state.input_counter += 1
    
    # Rerun to update the chat display
    st.rerun()

# Sidebar with additional options
with st.sidebar:
    st.header("Chat Options")
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This AI Knowledge Assistant uses multiple specialized agents to "
        "understand your intent, retrieve relevant information, analyze images,pdfs  "
        "and provide reasoned responses."
    )
    
    st.markdown("### Tips")
    st.markdown("""
    - Ask clear and specific questions
    - You can ask follow-up questions
    - The assistant remembers conversation context
    """)
    
    # Display message count
    if st.session_state.messages:
        st.markdown("---")
        st.metric("Total Messages", len(st.session_state.messages))