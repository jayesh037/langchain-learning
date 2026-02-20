from langchain_community.document_loaders import WebBaseLoader
import os
os.environ["USER_AGENT"] = "Mozilla/5.0"


url = 'https://en.wikipedia.org/wiki/Boot'
loader =  WebBaseLoader(url)

docs =loader.load()

print(len(docs))
print(docs[0].page_content)