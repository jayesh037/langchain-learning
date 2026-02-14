from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

schema =[
    ResponseSchema(name="fact1",description="Fact 1 about topic"),
    ResponseSchema(name="fact2",description="Fact 2 about topic"),
    ResponseSchema(name="fact3",description="Fact 3 about topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template =PromptTemplate(
    template="Give me 3 facts about {topic}.\n {format_instructions}",
    input_variables=['topic'],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain = template | model | parser

result= chain.invoke({'topic':'Sun'})
print(result)

# Cant execute as this version ie. 1.2.8 of langchain doesnt support structured output parser. It is only supported in the older version ie. 0.0.350 of langchain.