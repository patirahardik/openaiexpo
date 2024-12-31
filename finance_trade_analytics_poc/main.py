import os
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import streamlit as st

global vectorindex_openai

# loading envrionment variables
load_dotenv()

st.title("News Research Tool.")

#side bar
st.sidebar.title("New Articles")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Proccess URLs")
file_path = "faiss_store_openai.pkl"

main_placefolder = st.empty()

if process_url_clicked:
    # load the data
    loaders = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading Started....")
    data = loaders.load()

    # split data
    main_placefolder.text("Splitting Started....")
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )

    # As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
    docs = text_splitter.split_documents(data)

    # create embedding
    main_placefolder.text("Starting embedding....")
    # Create the embeddings of the chunks using openAIEmbeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    # Pass the documents and embeddings inorder to create FAISS vector index
    vectorindex_openai = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index to a pickle file
    vectorindex_openai.save_local("faiss_index")

    
query = main_placefolder.text_input("Question: ")
# Initialise LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=500) 
if query:
    if os.path.exists(file_path):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        new_vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=new_vector_store.as_retriever()) 
        chain_response = chain({"question": query})
        st.header("Answer")
        st.subheader(chain_response["answer"])

        # display sources, if available
        sources = chain_response.get("sources", "")
        if sources:
            st.subheader("Sources: ")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    else:
        st.subheader("No Query")
