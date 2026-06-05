import fitz
import json
import uuid
import base64
import ollama
import os
from app.core.config import settings
from app.core.database import get_db_connection
from app.core.storage import storage
# 🚨 新增：引入 LangChain 的高级递归分块器
from langchain_text_splitters import RecursiveCharacterTextSplitter

def merge_rects(rect_list, margin=15):
    """
    将距离接近的元素框合并成一个完整的大图表框。
    margin 控制了多远的元素会被判定为同一个图表。
    """
    if not rect_list:
        return []
    merged = True
    while merged:
        merged = False
        new_rects = []
        while rect_list:
            r = rect_list.pop(0)
            is_intersect = False
            for i, other in enumerate(new_rects):
                expanded_other = other + (-margin, -margin, margin, margin)
                if r.intersects(expanded_other):
                    new_rects[i] = r | other
                    is_intersect = True
                    merged = True
                    break
            if not is_intersect:
                new_rects.append(r)
        rect_list = new_rects
    return rect_list

def process_and_ingest_document(file_path: str, filename: str):
    """
    【架构解析：V5.0 高级入库流水线】
    阶段 1：使用递归滑动窗口进行文本切块 (Chunking & Overlap)
    阶段 2：逐页扫描视觉元素，使用 VLM 多模态解析架构图
    """
    print(f"\n⚙️ [业务层] 启动高级 RAG 解析流水线: {filename}")
    
    try:
        with fitz.open(file_path) as doc:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    
                    # ==========================================
                    # 🚀 阶段 1: 高级纯文本处理 (滑动窗口语义切分)
                    # ==========================================
                    print("📄 [1/2] 正在提取全篇文本并进行高级语义切分...")
                    
                    # 1. 实例化切分器：每次切 500 字，首尾重叠 100 字
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,     
                        chunk_overlap=100,  
                        # 优先按段落切，切不动再按句号，再切不动按逗号
                        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""] 
                    )
                    
                    # 2. 收集整本书的文本和对应的页码元数据
                    texts = []
                    metadatas = []
                    for page_num in range(len(doc)):
                        page_text = doc.load_page(page_num).get_text()
                        if len(page_text.strip()) > 50:
                            texts.append(page_text)
                            metadatas.append({"source": filename, "page": page_num + 1, "type": "text"})
                            
                    # 3. 核心魔法：切分时，LangChain 会自动把元数据映射到每一块小切片上！
                    if texts:
                        chunks = text_splitter.create_documents(texts, metadatas=metadatas)
                        print(f"✂️ 全文共被切分为 {len(chunks)} 个高质量语义区块。正在向量化入库...")
                        
                        for chunk in chunks:
                            # 为每个 500 字的切片生成高精度向量
                            vec = ollama.embeddings(model='bge-m3', prompt=chunk.page_content)['embedding']
                            cur.execute(
                                "INSERT INTO it_support_kb (content, metadata, embedding) VALUES (%s, %s, %s)",
                                (chunk.page_content, json.dumps(chunk.metadata), vec)
                            )
                        conn.commit() # 提交文本事务

                    # ==========================================
                    # 👁️ 阶段 2: 视觉元素扫描与 VLM 降噪打标
                    # ==========================================
                    print("👁️ [2/2] 正在启动多模态视觉神经元...")
                    for page_num in range(len(doc)):
                        print(f"📄 正在处理第 {page_num + 1}/{len(doc)} 页...")
                        page = doc.load_page(page_num)
                        page_rect = page.rect
                        
                        try:
                            # (这部分代码与你原先的图片处理逻辑完全一致)
                            raw_rects = []
                            for img_info in page.get_image_info(): raw_rects.append(fitz.Rect(img_info["bbox"]))
                            for path in page.get_drawings(): raw_rects.append(path["rect"])
                            for table in page.find_tables(): raw_rects.append(fitz.Rect(table.bbox))
                            
                            merged_charts = merge_rects(raw_rects, margin=20)
                            
                            valid_charts = []
                            for rect in merged_charts:
                                if rect.width < 100 or rect.height < 100: continue
                                if rect.width > page_rect.width * 0.9 and rect.height > page_rect.height * 0.9: continue
                                rect = rect + (-15, -15, 15, 15)
                                rect = rect.intersect(page_rect) 
                                valid_charts.append(rect)

                            for idx, chart_rect in enumerate(valid_charts):
                                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=chart_rect, alpha=False)
                                image_bytes = pix.tobytes("png")
                                
                                object_key = f"knowledge_images/{filename}_p{page_num+1}_chart{idx+1}_{uuid.uuid4().hex[:8]}.png"
                                storage.client.put_object(
                                    Bucket=settings.S3_BUCKET_NAME, Key=object_key, Body=image_bytes, ContentType='image/png'
                                )
                                cloud_url = f"{settings.MINIO_ENDPOINT}/{settings.S3_BUCKET_NAME}/{object_key}"
                                
                                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                                vlm_prompt = "你是资深架构师，请解析图表包含的组件及是否属于 Serverless，或指出其为非架构图。"
                                
                                try:
                                    print(f"   👁️ 正在呼叫 VLM 深度解析图表...")
                                    # 注：Ollama 原生包处理 Base64 图像比 LangChain 更稳定
                                    vlm_response = ollama.chat(
                                        model=settings.VLM_MODEL, 
                                        messages=[{'role': 'user', 'content': vlm_prompt, 'images': [base64_image]}]
                                    )
                                    desc = vlm_response['message']['content'].strip()
                                except Exception as e:
                                    desc = f"VLM 解析失败: {e}"
                                    print(f"   ⚠️ VLM 异常: {e}")

                                rich_content = f"【多模态图表资源】\n![架构图表]({cloud_url})\n\n【VLM 深度解析】\n{desc}"
                                img_vec = ollama.embeddings(model='bge-m3', prompt=rich_content)['embedding']
                                meta = {"source": filename, "page": page_num + 1, "type": "image", "url": cloud_url}
                                
                                cur.execute(
                                    "INSERT INTO it_support_kb (content, metadata, embedding) VALUES (%s, %s, %s)",
                                    (rich_content, json.dumps(meta), img_vec)
                                )
                            conn.commit()
                        except Exception as page_e:
                            print(f"   ⚠️ 第 {page_num + 1} 页图片处理异常，已回滚该页操作: {page_e}")
                            conn.rollback()
                            
        print(f"✅ 文件 {filename} 知识库入库彻底完成！")
        
    except Exception as main_e:
        print(f"❌ 解析流水线遭遇致命错误: {main_e}")
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ 已安全清理本地临时文件: {file_path}")