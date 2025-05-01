import json
from typing import Literal

from pydantic import BaseModel, StringConstraints

class HPRecommendation(BaseModel):
    recommendation: str = StringConstraints(max_length=32)
    importance_level: Literal[1,2,3,4,5]

class OutputSchema(BaseModel):
    summary: str
    hyperparameter_recommendations: dict[str, HPRecommendation]


print(json.dumps(OutputSchema.model_json_schema(), indent=2))
