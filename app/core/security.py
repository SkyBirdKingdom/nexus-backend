# app/core/security.py
from datetime import datetime, timedelta
import jwt
import bcrypt  # 🚨 引入原生 bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.core.config import settings
from app.core.database import get_db_connection
from cryptography.fernet import Fernet

# 告诉 FastAPI，登录接口的路由在哪里
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# 🚨 新增：初始化对称加密引擎
fernet_cipher = Fernet(settings.VAULT_SECRET_KEY.encode())

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    🚨 原生 bcrypt 校验密码：需要将字符串转为 bytes
    """
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )

def get_password_hash(password: str) -> str:
    """
    🚨 原生 bcrypt 生成哈希：自己生成盐并加密，最后转回字符串存入数据库
    """
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_bytes.decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    """FastAPI 全局鉴权拦截器"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="未授权的访问，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        if user_id is None:
            raise credentials_exception
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
                user = cur.fetchone()
                if not user:
                    raise credentials_exception
                    
        return {"user_id": user[0], "username": user[1]}
    except jwt.PyJWTError:
        raise credentials_exception

def encrypt_token(plain_token: str) -> str:
    """将外部系统明文 Token 加密为安全密文"""
    return fernet_cipher.encrypt(plain_token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    """Agent 取用时，将密文解密为明文 Token"""
    return fernet_cipher.decrypt(encrypted_token.encode()).decode()