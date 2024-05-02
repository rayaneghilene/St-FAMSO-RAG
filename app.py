from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

import streamlit as st
import ollama
import os
import json
# from numpy.linalg import norm
from PyPDF2 import PdfReader
from typing import Dict, Generator
from glob import glob



### LLM 🤓

def ollama_generator(messages: Dict) -> Generator:
    stream = ollama.chat(
            model="llava:7b",
            messages=messages,
            stream=True
        )
    for chunk in stream:
        yield chunk['message']['content']



### Data Integration
folder_path= '/Users/rayaneghilene/Documents/Ollama/RAG/Famso-Data'
@st.cache_resource
def load_pdf():
    # pdf_name ='Issues with Entailment-based Zero-shot Text Classification.pdf'
    #loaders = [PyPDFLoader(pdf_name)]
    pdf_files = glob(f"{folder_path}/*.pdf")
    loaders = [PyPDFLoader(file_path) for file_path in pdf_files]

    index= VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name= 'all-MiniLM-L12-V2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    return index

index = load_pdf()


### Chain

chain= RetrievalQA.from_chain_type(
    llm= ChatOllama(model="llava:7b"),
    chain_type ='stuff',
    retriever= index.vectorstore.as_retriever(),
    input_key='question'
)



### INTERFACE

with st.sidebar:
    # st.title('Side Bar')
    st.image('/Users/rayaneghilene/Documents/Ollama/RAG/famso_logo.png',  use_column_width='auto')
    # st.file_uploader('Upload your own file')
st.title('FAMSO Chatbot')
st.text('Supported class: Bactériologie- Pr Manel Marzouk-Dr Farah Azouzi')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Insert your text here :)')




if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    st.spinner(text='In progress')
    
    response = chain.run(prompt)
    st.chat_message("assistant").markdown(response)

    st.session_state.messages.append( {"role": "assistant", "content": response})    

    # with st.chat_message("assistant"):
    #     # response = st.write_stream(ollama_generator(st.session_state.messages))
    #     response = st.write_stream(chain.run(st.session_state.messages))
  
    # st.session_state.messages.append({"role": "assistant", "content": response})


    # with st.chat_message("assistant"):
    #     response = st.write_stream(ollama_generator(
    #         st.session_state.selected_model, st.session_state.messages)) 
    #     st.session_state.messages.append( {"role": "assistant", "content": response})

