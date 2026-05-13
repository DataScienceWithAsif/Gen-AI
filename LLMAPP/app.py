import streamlit as st
from dotenv import load_dotenv
import os

# Updated imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

os.environ["PINECONE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"]=""

from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


index=pc.Index("testing")


from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
llm=ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"],model_name="gpt-4o-mini-2024-07-18", temperature=.4)
chain=load_qa_chain(llm, chain_type="stuff")


load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

file_path = st.file_uploader("Upload your PDF", type=["pdf"])

if file_path:
    # Save uploaded file temporarily
    temp_path = os.path.join("temp", file_path.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(file_path.getbuffer())

    # Load single PDF file
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    st.write(chunks[0].page_content)  # Display first chunk
    
    docsearch=PineconeVectorStore.from_texts(
    [ch.page_content for ch in chunks],
    embeddings,
    index_name="testing"
    )
    
    query=st.text_input("Enter your query related Transformer....")
    
    match_results=docsearch.similarity_search(query)
    
    response=chain.run(input_documents=match_results, question=query)
    
    st.write(response)
    
    
    
