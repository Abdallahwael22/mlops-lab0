from control.Passenger import Passenger
from pydantic import BaseModel
from typing import List, Optional
class BatchPredictionRequest(BaseModel):
    passengers: List[Passenger]
    