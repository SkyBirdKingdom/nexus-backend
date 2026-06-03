# 封装流式输出的统一格式
import json
from typing import Any

def format_sse(status: str, **kwargs: Any) -> str:
    """
    【架构解析：协议层抽象】
    将业务字段组装成 Server-Sent Events 严格要求的 `data: JSON \n\n` 格式。
    未来如果需要增加事件类型 (event: ...)，只需修改这里。
    """
    payload = {"status": status}
    payload.update(kwargs)
    return f"data: {json.dumps(payload)}\n\n"