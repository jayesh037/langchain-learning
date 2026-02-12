from typing import TypedDict

class newperson(TypedDict):
    name: str
    age: int

per1=newperson(name="Alice", age=30)
print(per1)