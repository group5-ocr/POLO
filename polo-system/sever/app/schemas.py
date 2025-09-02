from pydantic import BaseModel
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class PaperOut(BaseModel):
    id: int
    filename: str
    result_path: str
    created_at: datetime

    class Config:
        orm_mode = True
