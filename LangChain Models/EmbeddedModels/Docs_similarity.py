from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = ["Virat Kohli is an Indian cricketer known for his aggressive batting and leadership. ",
        "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills." ,
        "Sachin Tendulkar, also known as the 'God of CriQet',holds many batting records." ,
        "Rohit Sharma is known for his elegant batting and record-breaking double centuries." ,
        "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]

query= "Tell me about Virat Kohli"

doc_embedding = embeddings.embed_documents(docs)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0] # 2d to 1d

index,score=sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(query)
print(docs[index])
print("Simmilarity Score: ", score)