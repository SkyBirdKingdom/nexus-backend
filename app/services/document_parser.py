import fitz
import json
import uuid
import base64
import ollama
import os
from collections import deque
from markitdown import MarkItDown
from app.core.config import settings
from app.core.database import get_db_connection
from app.core.storage import storage
# 🚨 新增：引入 LangChain 的高级递归分块器
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 🚨 新增：引入 Unstructured
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean_extra_whitespace

from bs4 import BeautifulSoup


# 🚨 新增：降维转换器 (HTML -> Markdown)
def html_table_to_markdown(html_string: str) -> str:
    """
    将带有繁重标签的 HTML 表格转换为极简的 Markdown 矩阵。
    彻底消灭检索时的字面量稀释惩罚。
    """
    if not html_string:
        return ""
    
    soup = BeautifulSoup(html_string, 'html.parser')
    rows = soup.find_all('tr')
    if not rows:
        return ""
        
    md_lines = []
    for i, row in enumerate(rows):
        # 提取表头(th)或单元格(td)
        cols = row.find_all(['th', 'td'])
        # 清理多余换行和空格，防止破坏 Markdown 结构
        col_texts = [clean_extra_whitespace(col.get_text()).replace('|', '-') for col in cols]
        
        md_lines.append("| " + " | ".join(col_texts) + " |")
        
        # 在第一行（表头）下方自动补全 Markdown 的分割线
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            
    return "\n".join(md_lines)

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

def process_pdf_with_vlm(file_path: str, filename: str):
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

# 🚨 2. 新增：Unstructured 工业级通用流水线
def process_unstructured_pipeline(file_path: str, filename: str):
    print(f"\n🏭 [业务层] 启动 Unstructured 多模态流水线: {filename}")
    
    img_temp_dir = f"temp_uploads/images_{uuid.uuid4().hex[:8]}"
    os.makedirs(img_temp_dir, exist_ok=True)
    
    try:
        elements = partition(
            filename=file_path, 
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            extract_image_block_output_dir=img_temp_dir
        )
        
        final_chunks = []
        current_text_buffer = ""
        
        # 🚨 核心修复 1：定义容量为 3 的滑动窗口，永远只记忆最近的 3 段有效文本
        context_window = deque(maxlen=3)
        last_page_num = "1"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,     
            chunk_overlap=150,  
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "]
        )

        # 闭包函数：将累积的纯文本切块并安全透传页码
        def flush_text_buffer(page):
            nonlocal current_text_buffer, final_chunks
            if current_text_buffer.strip():
                texts = text_splitter.create_documents(
                    [current_text_buffer], 
                    metadatas=[{"source": filename, "type": "unstructured_text", "page": page}]
                )
                final_chunks.extend(texts)
                current_text_buffer = ""

        for element in elements:
            # 🚨 核心修复 2：准确提取 Unstructured 识别到的原生页码
            page_num = getattr(element.metadata, "page_number", last_page_num)
            last_page_num = page_num # 更新缓存
            
            if element.category == "Table":
                flush_text_buffer(page_num)
                html_table = getattr(element.metadata, "text_as_html", element.text)

                context_str = " ".join(context_window) if context_window else "文档正文"
                
                # 🚨 核心改造 1：将繁重的 HTML 转换为轻量的 Markdown 矩阵
                markdown_table = html_table_to_markdown(html_table)
                
                # 🚨 核心改造 2：搜索面 (content) 使用纯净的 Markdown 格式
                # 向量模型对 Markdown 语法的行列对齐极为敏感，得分将暴涨！
                searchable_text = f"【表格关联上下文：{context_str}】\n{markdown_table}"
                
                # 🚨 核心改造 3：展示面 (metadata) 依然保留原汁原味的 HTML
                display_html = f"【表格关联上下文：{context_str}】\n<div class='table-wrapper'>\n{html_table}\n</div>\n"
                
                final_chunks.append(Document(
                    page_content=searchable_text, 
                    metadata={
                        "source": filename,
                        "type": "unstructured_table",
                        "page": page_num,
                        "original_html": display_html # 被藏在这里，待检索器狸猫换太子
                    }
                ))
                
            elif element.category == "Image":
                flush_text_buffer(page_num)
                img_path = getattr(element.metadata, "image_path", None)
                if img_path and os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        image_bytes = f.read()
                    
                    object_key = f"knowledge_images/{filename}_unstructured_{uuid.uuid4().hex[:8]}.png"
                    storage.client.put_object(
                        Bucket=settings.S3_BUCKET_NAME, Key=object_key, Body=image_bytes, ContentType='image/png'
                    )
                    cloud_url = f"{settings.MINIO_ENDPOINT}/{settings.S3_BUCKET_NAME}/{object_key}"
                    
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    # 🚨 核心修复 3：同样将滑动窗口里的文本喂给 Qwen-VL，让它看图时有语境！
                    context_str = " ".join(context_window) if context_window else "文档正文"
                    try:
                        vlm_response = ollama.chat(
                            model=settings.VLM_MODEL, 
                            messages=[{'role': 'user', 'content': f'结合上下文【{context_str}】，详细描述这张插图的核心数据。', 'images': [base64_image]}]
                        )
                        desc = vlm_response['message']['content'].strip()
                    except:
                        desc = "图表深度解析失败"
                    
                    img_md = f"【插图关联上下文：{context_str}】\n![【多模态视觉情报】{desc}]({cloud_url})\n"
                    final_chunks.append(Document(
                        page_content=img_md, 
                        metadata={"source": filename, "type": "unstructured_image", "url": cloud_url, "page": page_num}
                    ))
                    
            else:
                text = clean_extra_whitespace(element.text)
                if text:
                    # 🚨 核心修复 4：只要是有效的文本，就推进滑动窗口里留作记忆
                    context_window.append(text)
                    
                    # 原有的文本拼接逻辑保持不变
                    if element.category == "Title":
                        current_text_buffer += f"\n# {text}\n"
                    else:
                        current_text_buffer += text + "\n"
                    
        # 遍历结束后，清空最后一批文本
        flush_text_buffer(last_page_num)
        
        print(f"✂️ 文件被智能切分为 {len(final_chunks)} 个防破损区块。正在向量化入库...")
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for chunk in final_chunks:
                    vec = ollama.embeddings(model='bge-m3', prompt=chunk.page_content)['embedding']
                    cur.execute(
                        "INSERT INTO it_support_kb (content, metadata, embedding) VALUES (%s, %s, %s)",
                        (chunk.page_content, json.dumps(chunk.metadata), vec)
                    )
            conn.commit()
        
        print(f"✅ 文件 {filename} Unstructured 入库彻底完成！共 {len(final_chunks)} 个区块。")
    except Exception as e:
        print(f"❌ Unstructured 流水线遭遇错误: {e}")
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        import shutil
        if os.path.exists(img_temp_dir):
            shutil.rmtree(img_temp_dir)

# 🚨 3. 新增：全局调度器入口 (被 endpoints 调用)
def process_and_ingest_document(file_path: str, filename: str):
    """
    【架构解析：调度器模式 (Dispatcher Pattern)】
    统一的入口点。根据文件扩展名动态分发到不同的解析引擎。
    未来引入多租户权限时，可以在这里进行拦截和配额校验。
    """
    ext = filename.lower()
    if ext.endswith('.pdf'):
        # PDF 依然走拥有极致多模态理解能力的旧流水线
        process_pdf_with_vlm(file_path, filename)
    else:
        # 其他办公格式 (Word, Excel, PPT) 走工业级 Unstructured 流水线
        process_unstructured_pipeline(file_path, filename)