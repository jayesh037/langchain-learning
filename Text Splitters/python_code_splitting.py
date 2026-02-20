from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
    class NakliPromptTemplate(Runnable):
        def __init__(self,template,input_variables):
            self.template = template
            self.input_variables= input_variables

        def invoke(self,input_dict):
            return self.template.format(**input_dict)

        def format(self,input_dict):
            return self.template.format(**input_dict)

    class RunnableConnector(Runnable):

        def __init__(self,runnable_list):
            self.runnable_list = runnable_list

        def invoke(self,input_data):
            for runnable in self.runnable_list:
            input_data= runnable.invoke(input_data)

            return input_data
    
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

res=splitter.split_text(text)
print(len(res))
print(res[0])