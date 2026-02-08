from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

docs = ["Hello world", "This is a test document."]

res=embeddings.embed_documents(docs)

print(str(res))