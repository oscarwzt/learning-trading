import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import openai

# Set OpenAI API key
with open("api.txt", "r") as file:
    openai.api_key = file.read().strip()
# Load index


@st.cache_resource
def load_chat_engine():
    storageContext = StorageContext.from_defaults(
        persist_dir="tradingVolatilityDB")
    index = load_index_from_storage(storageContext)
    deepseek = Ollama(model="deepseek-r1:7b", request_timeout=120)
    return index.as_chat_engine(
        llm=deepseek,
        similarity_top_k=5,
        use_async=False,
        chat_mode="condense_question",
        streaming=True
    )


chat_engine = load_chat_engine()

# Streamlit UI
st.title("📈 Volatility RAG Chatbot")
st.markdown("Ask me anything about volatility, skew, structured products, etc.")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)

user_input = st.chat_input("Ask a question about volatility...")

if user_input:
    # Append user message
    st.session_state.chat_history.append(("user", user_input))

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_box = st.empty()
        partial_response = ""

        with st.spinner("Thinking..."):
            response = chat_engine.stream_chat(
                user_input)
            for chunk in response.response_gen:
                partial_response += chunk
                response_box.markdown(partial_response + "▌")

        response_box.markdown(partial_response)
        # 🔁 Append bot response
        st.session_state.chat_history.append(("assistant", partial_response))

if st.sidebar.button("🔁 Reset Chat"):
    st.session_state.chat_history = []
    chat_engine.reset()
