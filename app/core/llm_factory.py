# Qwen2.5-32B 和 Qwen2.5-VL 统一实例化工厂
from langchain_ollama import ChatOllama
from app.core.config import settings

class LLMFactory:
    """
    【架构解析：控制反转 (IoC) 的前置准备】
    通过工厂统一下发 LLM 实例。未来如果要增加鉴权、并发限流、请求打点日志，
    只需在工厂内部做 AOP (面向切面编程) 增强即可，业务代码完全无感。
    """
    
    @staticmethod
    def get_text_llm(temperature: float = 0.1):
        """获取纯文本大模型 (如 Qwen2.5-32b)"""
        return ChatOllama(
            model=settings.LLM_MODEL, 
            temperature=temperature
        )
    
    @staticmethod
    def get_vision_llm(temperature: float = 0.1):
        """获取多模态视觉大模型 (如 Qwen2.5-VL)"""
        return ChatOllama(
            model=settings.VLM_MODEL, 
            temperature=temperature
        )
