from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str = 'Jay' # default value 
    age : Optional[int] = None # Optional field, can be None or an integer
    email : EmailStr
    cgpa : float= Field(gt=0,lt=10,default=7, description="CGPA must be between 0 and 10")

new_stud = {"name":"Jayesh","email":"abc@gmail.com",'cgpa':5}
# new_stud={}

student= Student(**new_stud)

# print(student)
# print(student.name)

student_dict= dict(student)
print(student_dict['age'])

student_json= student.model_dump_json() # For JSON serialization

