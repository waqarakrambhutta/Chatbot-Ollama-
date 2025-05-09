from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 
import streamlit as st

import os
from dotenv import load_dotenv


load_dotenv()

## Langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

## Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","You are helpful assistant. Please respond to user queries."),
    ("user","Question:{question}")
])

## streamlit framework
st.title('Langchain demo with Ollama Gemma3')
input_text = st.text_input("Search the topic you want")

## Ollama :gemma3:1b LLM
llm = Ollama(model='gemma3:1b')
output_parser = StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))