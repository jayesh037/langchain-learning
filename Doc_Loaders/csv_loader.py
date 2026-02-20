from langchain_community.document_loaders import CSVLoader

loader =  CSVLoader(r'E:\AI\LangChain\Doc_Loaders\fact_bowling_summary.csv')

data = loader.load()

print(len(data))
print(data[0])