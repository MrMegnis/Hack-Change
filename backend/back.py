import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Literal
from fastapi.middleware.cors import CORSMiddleware
from model.model import transform_data, Model, load_data, index2methods
import pandas as pd


class Signatures(BaseModel):
    common: Dict[Literal["mobile", "web"], int]
    special: Dict[Literal["mobile", "web"], int]


class InputData(BaseModel):
    clientId: str
    organizationId: str
    segment: Literal["Малый бизнес", "Средний бизнес", "Крупный бизнес"]
    role: Literal["ЕИО", "Сотрудник"]
    organizations: int
    currentMethod: Literal["SMS", "PayControl", "КЭП на токене", "КЭП в приложении"]
    mobileApp: bool
    signatures: Signatures
    availableMethods: List[Literal["method1", "method2", "SMS", "PayControl", "КЭП на токене", "КЭП в приложении"]]
    claims: int


api = FastAPI()


api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000", "http://127.0.0.1:3000"],
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
    model.load_all("model/label_encoders.pkl", "model/scaler.pkl", "model/model.pkl")
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(dataframe)
    output = model.predict(dataframe)
    print(output)
    return {"response": f"{index2methods[output[0]]}"}


if __name__ == '__main__':
    uvicorn.run(api, host='localhost', port=8000)
