# 返回给前端的数据结构
from pydantic import BaseModel
from typing import Optional, Any

class BaseResponse(BaseModel):
    """
    【架构解析：统一信封结构 (Envelope Pattern)】
    企业内部通常会规定 API 必须有 code, message, data。
    这样前端解析时可以写全局的 Axios 拦截器，统一处理错误弹窗。
    """
    code: int = 200
    message: str = "success"
    data: Optional[Any] = None

class DocumentIngestResponse(BaseResponse):
    filename: str
    total_pages: int
    processed_images: int