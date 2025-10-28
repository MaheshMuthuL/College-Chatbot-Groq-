import streamlit as st
from model import get_qa_chain, query_chatbot
import time

# Page configuration
st.set_page_config(
    page_title="College Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar
with st.sidebar:
    st.title("🎓 College Chatbot")
    st.markdown("---")
    st.markdown("""
    ### About
    This chatbot uses:
    - **Web Scraping**: BeautifulSoup
    - **LLM**: Llama 3.1 (via Groq API) ⚡
    - **Framework**: LangChain
    - **Vector DB**: ChromaDB
    - **Embeddings**: Sentence Transformers
    
    ### Instructions
    1. Get free Groq API key
    2. Run `scraper.py` to scrape college website
    3. Run `ingest.py` to process documents
    4. Ask questions about your college!
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Model info
    st.markdown("---")
    st.markdown("### System Status")
    if st.session_state.qa_chain is not None:
        st.success("✅ Model Loaded")
    else:
        st.warning("⏳ Loading Model...")

# Main app
st.title("🎓 College Information Chatbot")
st.markdown("Ask me anything about the college!")

# Initialize QA chain (only once)
if st.session_state.qa_chain is None:
    with st.spinner("Loading model and knowledge base... This may take a minute..."):
        try:
            st.session_state.qa_chain = get_qa_chain()
            st.success("Model loaded successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.error("Please make sure:")
            st.error("1. You've added GROQ_API_KEY to your .env file")
            st.error("2. You've run `scraper.py` to scrape the website")
            st.error("3. You've run `ingest.py` to create the vector database")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about the college..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from chatbot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                # Query the chatbot
                answer, sources = query_chatbot(st.session_state.qa_chain, prompt)
                
                # Display answer
                full_response = answer
                message_placeholder.markdown(full_response)
                
                # Display sources if available
                if sources:
                    with st.expander("📚 View Sources"):
                        for idx, doc in enumerate(sources, 1):
                            st.markdown(f"**Source {idx}:**")
                            st.markdown(doc.page_content[:300] + "...")
                            st.markdown("---")
                
            except Exception as e:
                full_response = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Powered by BeautifulSoup 🥣 + Llama 🦙 + LangChain 🦜 + Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
