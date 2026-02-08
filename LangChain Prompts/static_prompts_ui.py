from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os   
import streamlit as st

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=512
)

model=ChatHuggingFace(llm=llm)

st.header("Research Assistant")

user_input= st.text_input("Enter your Promt")

if st.button("Generate Response"):
    result=model.invoke(user_input)
    st.write(result.content)

