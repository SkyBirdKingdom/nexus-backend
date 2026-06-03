# 前端请求的数据结构
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """
    【架构解析：请求数据契约】
    严格定义前端传过来的 JSON 格式。
    Field 里的描述和约束，不仅能自动生成 Swagger UI API 文档，
    还能在前端少传、传错字段时，自动返回极度清晰的 422 错误，省去了无数的 if-else 校验。
    """
    message: str = Field(..., description="用户的提问指令", min_length=1)
    thread_id: str = Field(default="default_session", description="会话标识，用于多轮对话记忆")