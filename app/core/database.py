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