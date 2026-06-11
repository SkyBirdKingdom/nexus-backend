# app/api/endpoints/credentials.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from app.core.security import get_current_user, encrypt_token
from app.core.database import get_db_connection
from app.schemas.responses import BaseResponse

router = APIRouter()

class CredentialBindRequest(BaseModel):
    platform: str  # e.g., "github", "lark"
    token: str     # 用户输入的明文 Token

@router.post("/bind", response_model=BaseResponse)
async def bind_credential(request: CredentialBindRequest, current_user: dict = Depends(get_current_user)):
    """绑定或更新外部系统的 API Token"""
    # 1. 在内存中极速加密 (绝不把明文落盘)
    safe_encrypted_token = encrypt_token(request.token)
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 2. 经典的 Upsert 语法 (存在则更新，不存在则插入)
            cur.execute("""
                INSERT INTO user_credentials (user_id, platform, encrypted_token, updated_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, platform) 
                DO UPDATE SET encrypted_token = EXCLUDED.encrypted_token, updated_at = CURRENT_TIMESTAMP
            """, (current_user["user_id"], request.platform, safe_encrypted_token))
        conn.commit()
        
    return BaseResponse(message=f"{request.platform.capitalize()} 联邦凭证已安全存入保险库。")

@router.get("/list", response_model=BaseResponse)
async def list_credentials(current_user: dict = Depends(get_current_user)):
    """获取当前用户已绑定的平台列表 (绝对脱敏，只返回名称)"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT platform, updated_at FROM user_credentials WHERE user_id = %s",
                (current_user["user_id"],)
            )
            rows = cur.fetchall()
            
            # 返回脱敏清单
            bound_platforms = [{"platform": r[0], "updated_at": r[1].strftime("%Y-%m-%d %H:%M")} for r in rows]
            
    return BaseResponse(data=bound_platforms)

@router.delete("/{platform}", response_model=BaseResponse)
async def unbind_credential(platform: str, current_user: dict = Depends(get_current_user)):
    """彻底销毁指定平台的访问凭证"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_credentials WHERE user_id = %s AND platform = %s",
                (current_user["user_id"], platform)
            )
        conn.commit()
    return BaseResponse(message=f"{platform.capitalize()} 凭证已从保险库中物理销毁。")