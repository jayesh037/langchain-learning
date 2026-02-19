from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1= PromptTemplate(
    template="Generate a tweet on {topic}",
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template="Generate a Linkedin post on {topic}",
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)
})

res=parallel_chain.invoke({'topic':'AI'})
print(res['tweet'])
print(res['linkedin'])