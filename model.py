from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
VECTOR_DB_PATH = "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq's Llama 3.1 model
TEMPERATURE = 0.3
TOP_K_RESULTS = 4

# Custom prompt template
PROMPT_TEMPLATE = """You are a helpful college assistant chatbot. Use the following pieces of context from the college information to answer the question at the end.
If you don't know the answer based on the provided context, just say that you don't have that information. Don't try to make up an answer.
Always be polite, clear, and helpful in your responses.

Context: {context}

Question: {question}

Helpful Answer:"""

def load_embeddings():
    """
    Load the embedding model.
    """
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def load_vector_store(embeddings):
    """
    Load the existing ChromaDB vector store.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        raise ValueError(f"Vector store not found at {VECTOR_DB_PATH}. Please run scraper.py and then ingest.py first.")
    
    print("Loading vector store...")
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="college_data"
    )
    return vector_store

def initialize_llm():
    """
    Initialize the Groq LLM via API.
    """
    print(f"Initializing {LLM_MODEL} via Groq API...")
    
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found! Please add it to your .env file")
    
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        groq_api_key=api_key,
        max_tokens=1024
    )
    return llm

def create_qa_chain(llm, vector_store):
    """
    Create a RetrievalQA chain with custom prompt.
    """
    print("Creating QA chain...")
    
    # Create custom prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def get_qa_chain():
    """
    Main function to initialize and return the QA chain.
    """
    try:
        # Load embeddings
        embeddings = load_embeddings()
        
        # Load vector store
        vector_store = load_vector_store(embeddings)
        
        # Initialize LLM
        llm = initialize_llm()
        
        # Create QA chain
        qa_chain = create_qa_chain(llm, vector_store)
        
        print("QA chain initialized successfully!")
        return qa_chain
    
    except Exception as e:
        print(f"Error initializing QA chain: {e}")
        raise

def query_chatbot(qa_chain, question):
    """
    Query the chatbot with a question.
    """
    try:
        result = qa_chain.invoke({"query": question})
        return result["result"], result.get("source_documents", [])
    except Exception as e:
        return f"Error: {str(e)}", []

# For testing the model directly
if __name__ == "__main__":
    print("Testing the chatbot model...")
    qa_chain = get_qa_chain()
    
    # Test query
    test_question = "What are the admission requirements?"
    print(f"\nQuestion: {test_question}")
    answer, sources = query_chatbot(qa_chain, test_question)
    print(f"\nAnswer: {answer}")
    print(f"\nNumber of sources: {len(sources)}")
