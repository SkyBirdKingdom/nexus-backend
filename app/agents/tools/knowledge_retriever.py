import psycopg2
import ollama
import json
from langchain_core.tools import tool
from sentence_transformers import CrossEncoder
from app.core.config import settings
from app.core.database import get_db_connection

# 全局单例重排引擎 (系统启动时加载到内存)
print("⏳ [系统初始化] 正在加载 Reranker 精度引擎...")
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
print("✅ Reranker 引擎点火完毕！")

@tool
def search_knowledge_base(query: str) -> str:
    """
    核心私有资产检索工具 (两段式混合检索)。
    """
    print(f"\n   [引擎轰鸣] 正在执行混合检索: '{query}'")
    
    # 1. 粗排阶段
    try:
        query_vector = ollama.embeddings(model='bge-m3', prompt=query)['embedding']
    except Exception as e:
        return f"向量化失败: {e}"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                sql = "SELECT content, metadata FROM it_support_kb ORDER BY embedding <=> %s::vector LIMIT 15"
                cur.execute(sql, (query_vector,))
                coarse_results = cur.fetchall()
            
            if not coarse_results:
                return "未检索到相关内容。"
            
            # 2. 精排阶段 (Reranker)
            sentence_pairs = [[query, row[0]] for row in coarse_results]
            scores = reranker.predict(sentence_pairs)
            scored_results = list(zip(scores, coarse_results))
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # 3. 熔断机制 (置信度底线)
            CONFIDENCE_THRESHOLD = 0.15 
            highest_score = scored_results[0][0]
            
            if highest_score < CONFIDENCE_THRESHOLD:
                return "【系统指令】：私有知识库中绝对没有关于此问题的信息，请诚实拒绝。"
                
            top_3_results = [res for res in scored_results[:3] if res[0] >= CONFIDENCE_THRESHOLD]
            
            # 4. 组装情报
            report_lines = [f"基于高精度检索，找到以下与 '{query}' 最相关的绝密依据：\n"]
            for rank, (score, row) in enumerate(top_3_results):
                content, metadata = row
                meta_dict = metadata if isinstance(metadata, dict) else json.loads(metadata)
                
                source = meta_dict.get('source', '未知文档')
                report_lines.append(f"### [置信度: {score:.2f} | 依据 {rank+1}] 来源: {source}")
                report_lines.append(f"内容片段: {content}\n")
                
            return "\n".join(report_lines)
            
    except Exception as e:
        return f"检索执行崩溃: {e}"