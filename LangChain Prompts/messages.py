from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=512
    )

model=ChatHuggingFace(llm=llm)  

message= [
    SystemMessage(content="You are a helpful assistant that provides concise and accurate information."),
    HumanMessage(content="What is the capital of France?")
]

result=model.invoke(message)
message.append(AIMessage(content=result.content))

print("Chat history:", message)