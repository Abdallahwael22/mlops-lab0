from pydantic import BaseModel
from typing import List, Optional   

class Passenger(BaseModel):
    Pclass: int
    Name: str
    Sex:str
    Parch: int
    Ticket: str
    Fare: float
    Cabin: Optional[str] = None
    Embarked: Optional[str]= None
    Age: Optional[float] = None
    SibSp: Optional[int] = None