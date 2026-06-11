# 环境变量与全局配置 (读取 .env)
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    【架构解析：强类型全局配置单例】
    继承自 BaseSettings 后，Pydantic 会自动去读取同目录下的 .env 文件，
    或者操作系统的环境变量。如果缺少必填项，服务在启动的第 0 秒就会报错崩溃 (Fail Fast)，
    这比运行到一半才报 "数据库连接失败" 要安全得多。
    """
    
    # 项目元数据
    PROJECT_NAME: str = "Nexus Agentic RAG"
    VERSION: str = "1.0.0"

    # PostgreSQL / pgvector 配置
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "your_password"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "postgres"

    # MinIO 对象存储配置
    MINIO_ENDPOINT: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "nexus-rag-assets"

    # 大模型配置
    LLM_MODEL: str = "qwen2.5:32b"
    VLM_MODEL: str = "qwen2.5-vl"
    
    JWT_SECRET_KEY: str = "your_jwt_secret_key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_DAYS: int = 7

    VAULT_SECRET_KEY: str = "your_vault_secret_key"

    class Config:
        env_file = ".env" # 指定环境变量文件路径

# 全局单例，整个项目只需导入这个 settings 对象即可
settings = Settings()