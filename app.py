# app.py
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ✅ CORRECT IMPORT (no .py)
from main import reg_simple, retriever, llm

# ────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Insight RAG • Asfand",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Clean Professional Styling
# ────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
h1, h2, h3 {
    color: #ffffff !important;
}
p, span, label, div {
    color: #e5e7eb !important;
}
.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}
.stChatMessage.user {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px;
}
.stChatMessage.assistant {
    background-color: #f9fafb !important;
    color: #111827 !important;
    border-radius: 10px;
}
textarea {
    background-color: #1f2937 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Session State
# ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Built by Asfand • {datetime.now().year}")
    st.caption("Quantum Insight RAG v1.0")

# ────────────────────────────────────────────────
# Main Chat UI
# ────────────────────────────────────────────────
st.title("⚛️ Quantum Computing RAG Assistant")
st.caption("Ask questions about quantum computing research papers.")

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
prompt = st.chat_input("Ask a quantum computing question...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from quantum knowledge base..."):
            answer = reg_simple(prompt, retriever, llm, top_k=3)

        st.markdown(answer)

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()