import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Literal
from fastapi.middleware.cors import CORSMiddleware
from model.model import transform_data, Model


class Signatures(BaseModel):
    common: Dict[Literal["mobile", "web"], int]
    special: Dict[Literal["mobile", "web"], int]


class InputData(BaseModel):
    clientId: str
    organizationId: str
    segment: Literal["Малый бизнес", "Средний бизнес", "Крупный бизнес"]
    role: Literal["ЕЮЛ", "Сотрудник"]
    organizations: int
    currentMethod: Literal["SMS", "PayControl", "КЭП на токене", "КЭП в приложении"]
    mobileApp: bool
    signatures: Signatures
    availableMethods: List[Literal["method1", "method2", "SMS", "PayControl", "КЭП на токене", "КЭП в приложении"]]
    claims: int


api = FastAPI()


api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.post('/predict')
async def predict_handler(datas: InputData):
    print(datas)
    print(os.path.abspath(__file__))
    dataframe = transform_data([datas.dict()])
    model = Model()
    model.load_model("../model/model.pkl")
    a = model.predict(dataframe)
    print(a)
    return {"numbers": f"{a}"}


if __name__ == '__main__':
    uvicorn.run(api, host='localhost', port=8000)
