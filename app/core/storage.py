# MinIO/S3 客户端封装
import boto3
from botocore.client import Config
import json
from app.core.config import settings

class MinioClient:
    """
    【架构解析：基础设施封装 (Infrastructure Encapsulation)】
    我们将 boto3 的具体实现隐藏在这个类里。
    对于上层的业务代码 (Agent) 来说，它不需要知道底层是 MinIO、AWS S3 还是阿里云 OSS，
    它只需要调用 `storage.upload_image()`。这就是面向接口编程的核心思想。
    """
    _instance = None # 单例指针

    def __new__(cls):
        # 【面试经典】：Python 实现单例模式的标准写法
        # 确保全局内存中只有一个 S3 客户端实例，节省巨大的网络建立开销
        if cls._instance is None:
            cls._instance = super(MinioClient, cls).__new__(cls)
            cls._instance.client = boto3.client(
                's3',
                endpoint_url=settings.MINIO_ENDPOINT,
                aws_access_key_id=settings.MINIO_ACCESS_KEY,
                aws_secret_access_key=settings.MINIO_SECRET_KEY,
                config=Config(signature_version='s3v4'),
                region_name='us-east-1'
            )
            cls._instance._ensure_bucket_exists()
        return cls._instance

    def _ensure_bucket_exists(self):
        """确保 Bucket 存在且拥有公共读权限"""
        try:
            self.client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
        except:
            print(f"🪣 Bucket {settings.S3_BUCKET_NAME} 不存在，正在创建...")
            self.client.create_bucket(Bucket=settings.S3_BUCKET_NAME)
            policy = {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Principal": "*", "Action": "s3:GetObject", "Resource": f"arn:aws:s3:::{settings.S3_BUCKET_NAME}/*"}]
            }
            self.client.put_bucket_policy(Bucket=settings.S3_BUCKET_NAME, Policy=json.dumps(policy))

# 实例化并暴露给外部使用
storage = MinioClient()