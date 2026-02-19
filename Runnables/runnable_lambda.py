from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

# def word_count(text):
#     return len(text.split())

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic']
)

joke_gen_chain= RunnableSequence(prompt,model,parser)

parallel_chain =  RunnableParallel({
    'joke': RunnablePassthrough(),
    # 'words':RunnableLambda(word_count)
    'words': RunnableLambda(lambda x:len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'Cricket'}))

