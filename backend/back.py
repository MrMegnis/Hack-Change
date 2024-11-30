import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Literal
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


@api.post('/predict')
async def predict_handler(
        datas: InputData
):
    dataframe = transform_data(datas.dict())
    model = Model()
    model.load_model()
    return {"number": f"{model.predict(dataframe)}"}


if __name__ == '__main__':
    uvicorn.run(api, host='localhost', port=8000)
