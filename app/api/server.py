from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import chat, documents, auth, credentials # 🚨 引入 auth
from app.core.database import init_db             # 🚨 引入 db init

# 启动时初始化表结构
init_db()

# 初始化 FastAPI 网关
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Multi-Agent RAG System API"
)

# 挂载业务路由 (所有 /chat 请求都会被转交到 chat.py 处理)
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(chat.router, prefix="/api/v1", tags=["Conversation"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Knowledge Base"])
app.include_router(credentials.router, prefix="/api/v1/credentials", tags=["Federated Vault"]) # 🚨 挂载路由

# 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """K8s 探针/负载均衡器专用的健康检查接口"""
    return {"status": "ok", "version": settings.VERSION}