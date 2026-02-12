from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated,Literal,Optional

model = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

class Review(TypedDict):
    # Annoated is used to add a description to the field, which can be helpful for documentation and understanding the purpose of the field.
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str,"A brief summary of the review"] 

    # sentiment: str (Its a simple TypedDict, so we can not add constraints like sentiment can only be positive, negative or neutral)

    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    #Literal is used to specify that the sentiment field can only take one of the specified values ("pos", "neg", "neutral"). This helps ensure that the output is consistent and adheres to the expected format.

    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

structured_model= model.with_structured_output(Review)

result= structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

                    The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

                    However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

                    Pros:
                    Insanely powerful processor (great for gaming and productivity)
                    Stunning 200MP camera with incredible zoom capabilities
                    Long battery life with fast charging
                    S-Pen support is unique and useful
                                                    
                    Cons:
                    Heavy and bulky design Bloatware in One UI Expensive price point Overall, the Samsung Galaxy S24 Ultra is a fantastic device for power users and photography enthusiasts, but it may not be the best choice for those who prioritize portability or are on a budget.   
                                                    
                                                    
                    Review by Jayesh SInghal""" )

print(result)
print(result['summary'])
print(result['pros'])
print(result['cons'])