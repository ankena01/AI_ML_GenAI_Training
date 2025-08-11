### Objective - Build a Simple LLM Application with LCEL (This is not using RAG) and inference it using LangServe 

### LangServe Documentation - https://python.langchain.com/docs/langserve/


import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes

### Load all the enviornment variables from the .env file
load_dotenv()

### Import Groq API KEY
groq_api_key = os.getenv('GROQ_API_KEY')

### Import the LLM model from Groq Cloud
model = ChatGroq(model='gemma2-9b-it' , groq_api_key = groq_api_key)



### Create the custom prompt template

prompt = ChatPromptTemplate.from_messages([
    ('system','Translate the following into {language}'),
    ('user','{text}')
])

### Create an output parser
output_parser = StrOutputParser()

### Create chains using LCEL
chain = prompt | model | output_parser

### Serve the LLM application 


### App definition

app = FastAPI(title = 'Simple LLM Application with LCEL (This is not using RAG)',
                version = '1.0',
                description = 'A simple API server using Langchain runnable interfaces')


### Adding chain routes
add_routes(app , chain , path='/chain')


### Entry Point

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app , host='127.0.0.1' , port = 8000)



