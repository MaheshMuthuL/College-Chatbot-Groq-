import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil

# Configuration
SCRAPED_DATA_PATH = "scraped_data"
VECTOR_DB_PATH = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    """
    Load documents from the scraped_data directory.
    """
    print("Loading scraped documents...")
    
    if not os.path.exists(SCRAPED_DATA_PATH):
        print(f"Error: {SCRAPED_DATA_PATH} directory not found!")
        print("Please run scraper.py first to scrape your college website.")
        return []
    
    # Load text files
    loader = DirectoryLoader(
        SCRAPED_DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )
    
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    """
    Split documents into smaller chunks for better retrieval.
    """
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """
    Create embeddings and store them in ChromaDB.
    """
    print("Creating embeddings and vector store...")
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Remove existing vector store if it exists
    if os.path.exists(VECTOR_DB_PATH):
        print("Removing existing vector store...")
        shutil.rmtree(VECTOR_DB_PATH)
    
    # Create new vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        collection_name="college_data"
    )
    
    print(f"Vector store created successfully at {VECTOR_DB_PATH}")
    return vector_store

def main():
    """
    Main ingestion pipeline.
    """
    print("Starting document ingestion pipeline...")
    print("="*80)
    
    # Load documents
    documents = load_documents()
    
    if len(documents) == 0:
        print("\nNo documents found!")
        print("Steps to fix:")
        print("1. Edit scraper.py and set your college website URL")
        print("2. Run: python scraper.py")
        print("3. Then run: python ingest.py")
        return
    
    # Split documents
    chunks = split_documents(documents)
    
    # Create vector store
    create_vector_store(chunks)
    
    print("\n" + "="*80)
    print("Ingestion completed successfully!")
    print(f"Your chatbot is ready with {len(chunks)} knowledge chunks!")

if __name__ == "__main__":
    main()
