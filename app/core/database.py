# PostgreSQL/pgvector 连接池与初始化
import psycopg2
from pgvector.psycopg2 import register_vector
from app.core.config import settings
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    """
    【架构解析：资源生命周期管理】
    这是一个标准的依赖注入 (Dependency Injection) 提供者。
    使用 contextmanager 模式，确保无论业务逻辑是否抛出异常，
    finally 块都会绝对执行，安全释放数据库连接。
    """
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT
        )
        # 确保该连接可以识别和操作 Vector 类型
        register_vector(conn)
        yield conn
    except psycopg2.Error as e:
        print(f"❌ 数据库连接致命错误: {e}")
        raise e
    finally:
        if conn is not None:
            conn.close()

def init_db():
    """
    【架构解析：启动断言】
    服务器启动时，强制建表并检查核心字段，防止业务运行到一半才报 SQL 错误。
    """
    print("⏳ [数据库] 正在校验表结构...")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            
            # 创建用户表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(50) PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    hashed_password VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建带有 user_id 强隔离的知识库表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS it_support_kb (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(1024),
                    user_id VARCHAR(50) NOT NULL  -- 🚨 核心隔离字段
                )
            """)
        conn.commit()
    print("✅ [数据库] 引擎校验完毕！")