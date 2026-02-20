from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

loader = DirectoryLoader(
    path=r"E:\AI\LangChain\Doc_Loaders\books",
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs=loader.load()
docs=loader.lazy_load()
print(len(docs))