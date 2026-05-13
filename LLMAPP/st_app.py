import streamlit as st
from dotenv import load_dotenv
import os
from pinecone import Pinecone

# LangChain imports (updated)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

os.environ["PINECONE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"]=""

# Set up API keys
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "testing"  # make sure index exists in Pinecone

# Initialize Embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name="gpt-4o-mini-2024-07-18",
    temperature=0.4
)
chain = load_qa_chain(llm, chain_type="stuff")


# ---- Helper Functions ---- #
def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return path."""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_pdf_to_chunks(file_path):
    """Load PDF and split into text chunks."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    return chunks


def store_chunks_in_pinecone(chunks):
    """Store text chunks in Pinecone vector DB."""
    PineconeVectorStore.from_texts(
        [ch.page_content for ch in chunks],
        embeddings,
        index_name=INDEX_NAME
    )


def query_pinecone_and_llm(query):
    """Search in Pinecone and get answer from LLM."""
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    match_results = docsearch.similarity_search(query)
    response = chain.run(input_documents=match_results, question=query)
    return response


# ---- Streamlit UI ---- #
st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("📄 PDF → Pinecone → LLM Q&A")

st.sidebar.header("📂 Upload & Process")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Step 1: Save file
        file_path = save_uploaded_file(uploaded_file)

        # Step 2: Create chunks
        chunks = process_pdf_to_chunks(file_path)
        st.success(f"✅ PDF processed into {len(chunks)} chunks.")

        # Step 3: Store in Pinecone
        store_chunks_in_pinecone(chunks)
        st.success("📥 Data saved in Pinecone successfully.")

    # Step 4: Ask questions
    st.subheader("💬 Ask a Question")
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Searching and generating answer..."):
            answer = query_pinecone_and_llm(query)
            st.write("**Answer:**", answer)
