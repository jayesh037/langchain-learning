from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id=os.getenv("repo_id1"),
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512
)

model=ChatHuggingFace(llm=llm)

try:
    result=model.invoke("What is capital of INDIA?")
    print(result.content)
except Exception as e:
    print(f"An error occurred: {e}")