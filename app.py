from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

import streamlit as st


# from wastonxlangchain import LangchainInterference


import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm
from PyPDF2 import PdfReader
from typing import Dict, Generator




def extract_text_from_pdf(pdf_path, output_txt_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        text = ''
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return(text)


def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)


def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    # save embeddings
    save_embeddings(filename, embeddings)
    return embeddings



### LLM 🤓

def ollama_generator(messages: Dict) -> Generator:
    stream = ollama.chat(
            model="llava:7b",
            messages=messages,
             stream=True
        )
    for chunk in stream:
        yield chunk['message']['content']


### INTERFACE
st.title('Ask Away....')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Insert your text here :)')

if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    st.spinner(text='In progress')
    


    with st.chat_message("assistant"):
        response = st.write_stream(ollama_generator(st.session_state.messages))

  
    st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.title('Side Bar')

    st.multiselect('Multiselect', ['llama:7b','llava:7b','mistral:7b'])
