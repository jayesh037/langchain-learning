# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI(model='openai/gpt-oss-120b:free', openai_api_base="https://openrouter.ai/api/v1")

# result = model.invoke("What is capital of INDIA?")
# print(result.content)

import os
print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
