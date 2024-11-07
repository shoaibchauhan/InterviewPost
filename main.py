import os
import django


os.environ.setdefault("DJANGO_SETTINGS_MODULE","InterviewPost.settings")

django.setup()
from fastapi import FastAPI
from myapp.router import router






app=FastAPI()
app.include_router(router)

@app.get("/")
def root():
    return {"Hello  USe this link for swagger":"http://127.0.0.1:8000/docs"}







