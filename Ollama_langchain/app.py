#####----------------------------------------------####

### Objective - To create a GEN AI Chatbot using OpenSource LLM model using Ollama



### Reference Documentation 
### Ollama on GitHub  https://github.com/ollama/ollama  
### (Ollama for Windows) https://github.com/ollama/ollama/blob/main/docs/windows.md
### Download Ollama on Windows - https://ollama.com/download/windows

import os


### Importing the enviornment variables for LANGSMITH tracking
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'




from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate       ### Create Custom ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

#### ------------------- Step 1 - Design the Prompt Template  ------------------- ####


### ChatPromptTemplate documentation - https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/

prompt = ChatPromptTemplate.from_messages([
    ('system' , 'You are a helpful AI Assistant. Please respond to the question asked'),
    ('user' , 'Question: {question}')
])


#### ------------------- Step 2 - Design the Streamlit Framework  ------------------- ####

st.title('Langchain demo with Ollama (llama3.2:1b) model')
input_text = st.text_input('Enter the question...')


#### ------------------- Step 3 - Intialize the LLM Model  ------------------- ####

llm = Ollama(model = 'llama3.2:1b')


#### ------------------- Step 4 - Create the String Output Parser  ------------------- ####

ouput_parser = StrOutputParser()


#### ------------------- Step 5 - Intialize the Chain  ------------------- ####

chain = prompt | llm | ouput_parser


#### ------------------- Step 6 - Finalize the flow of execution in Streamlit UI  ------------------- ####

if input_text:
    st.write(chain.invoke({'question' : input_text}))

