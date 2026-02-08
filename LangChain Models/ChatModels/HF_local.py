from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm=HuggingFacePipeline.from_model_id(model_id="meta-llama/Llama-3.2-1B-Instruct", task="text-generation", pipeline_kwargs={temperature:0.7, max_new_tokens:100})
model = ChatHuggingFace(llm=llm)

model.invoke()

result = model.invoke("What is capital of INDIA?")
print(result.content)