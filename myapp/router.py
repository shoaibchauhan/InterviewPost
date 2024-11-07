import uuid
from typing import List

from fastapi import APIRouter, UploadFile, File

from myapp import models
from myapp.controller import authenticate, check_file_extension, check_if_file_already_uploaded, \
    save_file_to_storage_folder, extract_text_from_file, generate_file_summary
from myapp.models import Filee
from myapp.schema import FileSummaryOut, FileOut

router=APIRouter()
from fastapi.security import  HTTPBasicCredentials, HTTPBasic
from fastapi import Depends, HTTPException

security = HTTPBasic()

#get list of files
@router.get("/v1/files",response_model=List[FileOut])
def list_files(credentials: HTTPBasicCredentials = Depends(security)):
    authenticate(credentials)
    files = list(models.Filee.objects.all().values('id', 'filename','filesummary'))
    if not files:
        return {"message": "No files found"}
    return files



#get file summary by file id
@router.get("/v1/files/{file_id}", response_model=FileSummaryOut)
def get_file_summary(file_id: int):
    # Attempt to fetch the file by its integer ID
    file = Filee.objects.filter(id=file_id).first()

    if file:
        return file
    else:
        raise HTTPException(status_code=404, detail="File not found")

# #get file summary by file id
# @router.get("/v1/files/{file_id}", response_model=FileSummaryOut)
# def get_file_summary(file_id: str, credentials: HTTPBasicCredentials = Depends(security)):
#     authenticate(credentials)
#     try:
#         file_id_uuid = uuid.UUID(file_id)
#         file = models.File.objects.filter(id=file_id_uuid).first()
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid file ID")
#
#     if file:
#         return file
#     else:
#         raise HTTPException(status_code=404, detail="File not found")

@router.post("/v1/files")
def upload_file(file: UploadFile = File(...)):
    check_file_extension(file)
    check_if_file_already_uploaded(file)
    file_path = save_file_to_storage_folder(file)
    file_extension = file.filename.split(".")[-1]
    text = extract_text_from_file(file_path, file_extension)
    summary = generate_file_summary(text)
    new_file = Filee.objects.create(filename=file.filename, filesummary=summary)
    return {"message": "File summary generated successfully","file_id": new_file.id}
