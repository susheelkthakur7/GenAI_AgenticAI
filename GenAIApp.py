#This is a demo to build a simple Gen AI App 
#LLM used - gemma:2b & streamlit for quick development
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
#print("API KEY:", os.getenv("LANGCHAIN_API_KEY"))
#print("PROJECT:", os.getenv("LANGCHAIN_PROJECT"))

os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']='TRUE'
os.environ['LANGCHAIN_PROJECT']=os.getenv("LANGCHAIN_PROJECT")
#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the questions asked"),
        ("user","Question:{question}")
    ]
)
##Streamlit framework
st.title("Langchain Ollama (gemma2b) Demo")
input_text=st.text_input("What questions do you have in mind?")
#llm  -gemma:2b
llm=OllamaLLM(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser
if input_text:
    response=chain.invoke({"question":input_text})
    st.write(response)
