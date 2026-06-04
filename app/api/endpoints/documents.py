from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from app.services.document_parser import process_and_ingest_document
from app.schemas.responses import BaseResponse
import os
import shutil

router = APIRouter()

# 定义临时文件存放目录
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=BaseResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    【架构解析：异步解耦上传接口】
    1. 接收前端传来的物理文件，并闪电般落盘到本地临时目录。
    2. 将漫长的多模态解析任务（process_and_ingest_document）挂载到 FastAPI 的后台任务池。
    3. 立刻切断 HTTP 连接，响应前端，绝不阻塞！
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="目前仅支持 PDF 格式文档的高精度解析")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # 将上传的文件流安全写入本地临时目录
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 🚨 核心逻辑：丢给后台任务，立刻返回！
    background_tasks.add_task(process_and_ingest_document, file_path, file.filename)
    
    return BaseResponse(
        code=200, 
        message="文件已接收！Nexus 引擎正在后台进行 VLM 多模态深度解析与向量化，请稍后向 Agent 提问验证。", 
        data={"filename": file.filename}
    )