from langchain_community.document_loaders import PyPDFLoader

loader =  PyPDFLoader(r'E:\AI\LangChain\Doc_Loaders\research.pdf')

docs =loader.load()

print(len(docs))