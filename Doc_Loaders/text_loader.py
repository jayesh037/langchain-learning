from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

loader = TextLoader(r"E:\AI\LangChain\Doc_Loaders\cricket.txt", encoding='utf-8')

docs= loader.load()

# print(docs[0])

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary for following report \n {text}",
    input_variables=['text']
)

chain = prompt | model | parser
print(chain.invoke({'text':docs[0].page_content}))