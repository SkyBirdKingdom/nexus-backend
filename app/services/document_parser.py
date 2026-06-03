# PDF 截取、VLM 打标、入库的核心逻辑
import fitz
import json
import uuid
import base64
import ollama
from app.core.config import settings
from app.core.database import get_db_connection
from app.core.storage import storage

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
    这里封装了最高复杂度的多模态入库流水线。
    它不关心请求是怎么进来的 (API/CLI)，只关心如何把数据处理好。
    """
    print(f"\n⚙️ [业务层] 启动 VLM 多模态解析流水线: {filename}")
    
    # 从依赖提供者获取连接
    db_gen = get_db_connection()
    conn = next(db_gen)
    
    try:
        doc = fitz.open(file_path)
        with conn.cursor() as cur:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_rect = page.rect
                
                try:
                    # 1. 文本处理 (与你之前代码一致)
                    text_content = page.get_text()
                    if len(text_content.strip()) > 50:
                        vec = ollama.embeddings(model='bge-m3', prompt=text_content)['embedding']
                        metadata = {"source": filename, "page": page_num + 1, "type": "text"}
                        cur.execute(
                            "INSERT INTO it_support_kb (content, metadata, embedding) VALUES (%s, %s, %s)",
                            (text_content, json.dumps(metadata), vec)
                        )
                    
                    # 2. 视觉元素扫描
                    raw_rects = []
                    for img_info in page.get_image_info(): raw_rects.append(fitz.Rect(img_info["bbox"]))
                    for path in page.get_drawings(): raw_rects.append(path["rect"])
                    for table in page.find_tables(): raw_rects.append(fitz.Rect(table.bbox))
                    
                    merged_charts = merge_rects(raw_rects, margin=20)
                    # ... 过滤降噪逻辑 ...
                    valid_charts = []
                    for rect in merged_charts:
                        if rect.width < 100 or rect.height < 100: continue
                        if rect.width > page_rect.width * 0.9 and rect.height > page_rect.height * 0.9: continue
                        rect = rect + (-15, -15, 15, 15)
                        rect = rect.intersect(page_rect) 
                        valid_charts.append(rect)

                    # 3. 传 MinIO 与 VLM
                    for idx, chart_rect in enumerate(valid_charts):
                        if chart_rect.width < 100 or chart_rect.height < 100: continue
                        
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
                            # 这里可以直接使用我们刚才写的 LLMFactory
                            vlm_response = ollama.chat(
                                model=settings.VLM_MODEL, 
                                messages=[{'role': 'user', 'content': vlm_prompt, 'images': [base64_image]}]
                            )
                            desc = vlm_response['message']['content'].strip()
                        except Exception as e:
                            desc = f"解析失败: {e}"

                        rich_content = f"【多模态图表资源】\n![架构图表]({cloud_url})\n\n【VLM 深度解析】\n{desc}"
                        img_vec = ollama.embeddings(model='bge-m3', prompt=rich_content)['embedding']
                        meta = {"source": filename, "page": page_num + 1, "type": "image", "url": cloud_url}
                        
                        cur.execute(
                            "INSERT INTO it_support_kb (content, metadata, embedding) VALUES (%s, %s, %s)",
                            (rich_content, json.dumps(meta), img_vec)
                        )
                    conn.commit()
                except Exception as e:
                    conn.rollback()
    finally:
        db_gen.close() # 触发 finally 关闭连接