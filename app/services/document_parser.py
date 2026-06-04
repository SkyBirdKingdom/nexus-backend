# PDF 截取、VLM 打标、入库的核心逻辑
import fitz
import json
import uuid
import base64
import ollama
from app.core.config import settings
from app.core.database import get_db_connection
from app.core.storage import storage
import os

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
                # 稍微扩大一点判定范围，防止靠得近但没相交的元素被切开
                expanded_other = other + (-margin, -margin, margin, margin)
                if r.intersects(expanded_other):
                    new_rects[i] = r | other  # 取并集 (Union)
                    is_intersect = True
                    merged = True
                    break
            if not is_intersect:
                new_rects.append(r)
        rect_list = new_rects
    return rect_list

def process_and_ingest_document(file_path: str, filename: str):
    """
    【架构解析：胖服务层 (Fat Service)】
    封装最高复杂度的多模态入库流水线。自带页级微事务与自动内存释放。
    """
    print(f"\n⚙️ [业务层] 启动 VLM 多模态解析流水线: {filename}")
    
    # 最外层异常捕获：保证系统不会因为某一个坏文件而崩溃
    try:
        # 使用 with 上下文管理器，确保 PDF 解析完后自动释放内存
        with fitz.open(file_path) as doc:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    for page_num in range(len(doc)):
                        print(f"📄 正在处理第 {page_num + 1}/{len(doc)} 页...")
                        page = doc.load_page(page_num)
                        page_rect = page.rect
                        
                        # 页级微事务异常捕获：某一页报错，只回滚当前页
                        try:
                            # ==========================================
                            # 1. 文本处理
                            # ==========================================
                            text_content = page.get_text()
                            if len(text_content.strip()) > 50:
                                vec = ollama.embeddings(model='bge-m3', prompt=text_content)['embedding']
                                metadata = {"source": filename, "page": page_num + 1, "type": "text"}
                                cur.execute(
                                    "INSERT INTO it_support_kb (content, metadata, embedding) VALUES (%s, %s, %s)",
                                    (text_content, json.dumps(metadata), vec)
                                )
                            
                            # ==========================================
                            # 2. 视觉元素扫描与降噪
                            # ==========================================
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

                            # ==========================================
                            # 3. 传 MinIO 与 VLM 打标
                            # ==========================================
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
                            
                            # 当前页解析全部成功，提交微事务
                            conn.commit()
                            
                        except Exception as page_e:
                            print(f"   ⚠️ 第 {page_num + 1} 页处理异常，已回滚该页操作: {page_e}")
                            conn.rollback() # 单页失败不影响其他页
                            
        print(f"✅ 文件 {filename} 知识库入库彻底完成！")
        
    except Exception as main_e:
        print(f"❌ 解析流水线遭遇致命错误: {main_e}")
        
    finally:
        # 【架构级优化】：解析完成后，自动清理 FastAPI 下载到硬盘的临时文件，防止磁盘写满
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ 已安全清理本地临时文件: {file_path}")