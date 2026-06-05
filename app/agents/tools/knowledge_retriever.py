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
    【架构解析：V5.0 混合检索】
    执行 Vector(语义) + Trigram(字面) 双路召回，通过 RRF 算法融合后，再交由 CrossEncoder 精排。
    """
    print(f"\n   [引擎轰鸣] 正在执行双路混合检索: '{query}'")
    
    # 1. 生成查询向量
    try:
        query_vector = ollama.embeddings(model='bge-m3', prompt=query)['embedding']
    except Exception as e:
        return f"向量化失败: {e}"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # 确保安装了 pgvector (向量检索) 和 pg_trgm (模糊全文检索)
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                
                # ==========================================
                # 🚨 核心黑魔法：在 SQL 层并发执行双路召回与 RRF 融合
                # ==========================================
                sql = """
                WITH vector_search AS (
                    -- 第一路：纯语义向量检索
                    SELECT content, metadata,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
                    FROM it_support_kb
                    LIMIT 20
                ),
                keyword_search AS (
                    -- 第二路：字面相似度检索 (利用 pg_trgm)
                    SELECT content, metadata,
                           ROW_NUMBER() OVER (ORDER BY similarity(content, %s) DESC) AS rank
                    FROM it_support_kb
                    WHERE content ILIKE %s OR similarity(content, %s) > 0.05
                    LIMIT 20
                )
                -- RRF 融合计算 (常数 k 通常取 60)
                SELECT
                    COALESCE(v.content, k.content) AS content,
                    COALESCE(v.metadata, k.metadata) AS metadata,
                    (COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0)) AS rrf_score
                FROM vector_search v
                FULL OUTER JOIN keyword_search k ON v.content = k.content
                ORDER BY rrf_score DESC
                LIMIT 15;
                """
                
                # ILIKE 用于绝对字面包含，similarity 用于近似拼写
                like_query = f"%{query}%"
                # 传入四次参数，分别对应 SQL 中的四个 %s
                cur.execute(sql, (query_vector, query, like_query, query))
                coarse_results = cur.fetchall()
            
            if not coarse_results:
                return "未检索到相关内容。"
            
            # ==========================================
            # 2. 精排阶段 (Reranker 交叉编码器)
            # ==========================================
            # coarse_results 的结构现在是: (content, metadata, rrf_score)
            sentence_pairs = [[query, row[0]] for row in coarse_results]
            scores = reranker.predict(sentence_pairs)
            scored_results = list(zip(scores, coarse_results))
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # 3. 熔断机制 (防幻觉底线)
            CONFIDENCE_THRESHOLD = 0.15 
            highest_score = scored_results[0][0]
            
            if highest_score < CONFIDENCE_THRESHOLD:
                return json.dumps([{"title": "系统拦截", "content": "知识库中绝对没有关于此问题的信息，请诚实拒绝。"}], ensure_ascii=False)
                
            top_3_results = [res for res in scored_results[:3] if res[0] >= CONFIDENCE_THRESHOLD]
            
            # 4. 🚨 核心改造：组装为独立的 JSON 数组，替代原来的字符串拼接
            results = []
            for rank, (rerank_score, row) in enumerate(top_3_results):
                content, metadata, rrf_score = row
                meta_dict = metadata if isinstance(metadata, dict) else json.loads(metadata)
                
                source = meta_dict.get('source', '未知文档')
                page = meta_dict.get('page', 'N/A')
                
                results.append({
                    "title": f"{source} (第{page}页)",
                    "content": content,
                    "rerank_score": round(float(rerank_score), 3),
                    "rrf_score": round(float(rrf_score), 3)
                })
                
            # 将数组转为 JSON 字符串交还给 Agent
            return json.dumps(results, ensure_ascii=False)
            
    except Exception as e:
        return f"检索执行崩溃: {e}"