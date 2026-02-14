from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


# 1st prompt -> Detailed report
template1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> Just a summary
template2=PromptTemplate(
    template="Write a 5 line summary on provided text. /n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'Black holes'})

result= model.invoke(prompt1)

prompt2= template2.invoke({'text':result})

final_result= model.invoke(prompt2)

print(final_result)