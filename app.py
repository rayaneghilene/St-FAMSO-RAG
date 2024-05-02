import streamlit as st
import time
import os
import json
import numpy as np
from numpy.linalg import norm
from PyPDF2 import PdfReader
from typing import Dict, Generator




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
    
    #st.spinner(text='In progress')
    


    #with st.chat_message("assistant"):
    #    response = st.write_stream(ollama_generator(st.session_state.messages))

    #st.session_state.messages.append({"role": "assistant", "content": response})


with st.sidebar:
    st.title('Side Bar')

    st.multiselect('Choose the model', ['llama:7b','llava:7b','mistral:7b'])
