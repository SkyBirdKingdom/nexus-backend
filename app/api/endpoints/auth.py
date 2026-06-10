# app/api/endpoints/auth.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import uuid

from app.core.security import verify_password, get_password_hash, create_access_token
from app.core.database import get_db_connection
from app.schemas.responses import BaseResponse

router = APIRouter()

class RegisterRequest(BaseModel):
    username: str
    password: str

@router.post("/register")
async def register(request: RegisterRequest):
    user_id = "U_" + uuid.uuid4().hex[:8]
    hashed_pwd = get_password_hash(request.password)
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (request.username,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="用户名已存在")
            
            cur.execute(
                "INSERT INTO users (id, username, hashed_password) VALUES (%s, %s, %s)",
                (user_id, request.username, hashed_pwd)
            )
        conn.commit()
        
    return BaseResponse(message="注册成功，请去登录。")

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    【架构解析：OAuth2 换票】
    注意：为了兼容 Swagger UI 和标准 OAuth2 协议，表单必须使用 OAuth2PasswordRequestForm。
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, hashed_password FROM users WHERE username = %s", (form_data.username,))
            user = cur.fetchone()
            
    if not user or not verify_password(form_data.password, user[2]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
        
    # 签发 JWT
    access_token = create_access_token(data={"sub": user[0], "username": user[1]})
    
    return {"access_token": access_token, "token_type": "bearer"}