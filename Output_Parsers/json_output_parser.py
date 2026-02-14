from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)
parser= JsonOutputParser()

template=PromptTemplate(
    template="Give me name, age and email of a fictional student. \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

# prompt = template.format()
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser #another way bt using chain instaed of invoking each separately
final_result = chain.invoke({})

print(final_result)
# print(type(final_result))