from uuid import UUID

from pydantic import BaseModel


class FileOut(BaseModel):
    id:int
    filename:str
    filesummary:str

    class Config:
        orm_mode = True  # Ensure FastAPI uses Django ORM models correctly

class FileSummaryOut(BaseModel):
    filename:str
    filesummary:str


