import streamlit as st
from agentstock import agent  # 👈 Import your existing agent object

st.set_page_config(page_title="Stock Analysis Chatbot", layout="centered")

st.title("📈 Stock Analysis Chatbot")
st.markdown("Ask anything about stock indicators like RSI, MACD, Supertrend, OBV, or ML signals.")

# Input box
query = st.text_input("🧠 Enter your question:", placeholder="e.g., What is the RSI for SBIN.NS on 2024-04-15?")

# Chat history display
if "messages" not in st.session_state:
    st.session_state.messages = []

# Run agent on query
if query:
    with st.spinner("🤖 Thinking..."):
        response = agent.run(query)

    # Save to chat history
    st.session_state.messages.append(("You", query))
    st.session_state.messages.append(("Bot", response))

# Show chat history
for sender, message in reversed(st.session_state.messages):
    st.markdown(f"**{sender}:** {message}")
