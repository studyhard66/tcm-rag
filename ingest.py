#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest.py — 中医知识库入库（RAG 第 1 步）
流程：读取 data/docs 下 Markdown/TXT → 按标题结构切分 → bge 向量化 → Chroma 持久化
重复运行会先删后建集合，避免数据重复。
"""
import os
import re
import glob
import warnings

import torch
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(ROOT, "data", "docs")
DB_DIR = os.path.join(ROOT, "vector_store")
COLLECTION_NAME = "tcm_docs"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
MAX_CHARS = 450   # 单块最大字数，适配 bge 的 512 token 上限
OVERLAP = 50      # 超长块二次切分时的重叠字数，避免割断语义

def split_markdown(text):
    """按一级/二级标题切块，块首附带'一级标题 > 二级标题'路径，提升检索质量。"""
    h1, h2, buf, chunks = "", "", [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            path = " > ".join(x for x in [h1, h2] if x)
            chunks.append((f"{path}\n{body}", path))

    for line in text.splitlines():
        m2 = re.match(r"^##\s+(.+)$", line)
        m1 = re.match(r"^#\s+(.+)$", line)
        if m2:                      # 二级标题 = 一个新块的开始
            flush(); buf.clear()
            h2 = m2.group(1).strip()
            buf.append(line)
        elif m1:
            flush(); buf.clear()
            h1, h2 = m1.group(1).strip(), ""
            buf.append(line)
        else:
            buf.append(line)
    flush()

    # 超长块滑窗二次切分
    final = []
    for content, path in chunks:
        if len(content) <= MAX_CHARS:
            final.append((content, path))
        else:
            for start in range(0, len(content), MAX_CHARS - OVERLAP):
                final.append((content[start:start + MAX_CHARS], path))
    return final

def split_plaintext(text, source):
    return [(text[i:i + MAX_CHARS], source)
            for i in range(0, len(text), MAX_CHARS - OVERLAP)]


_t2s = None


def to_simplified(text):
    """繁体转简体（OpenCC）；未安装 opencc 时原样返回并仅提示一次"""
    global _t2s
    if _t2s is None:
        try:
            from opencc import OpenCC
            _t2s = OpenCC("t2s")
        except Exception:
            print("      [警告] 未安装 opencc-python-reimplemented，跳过繁转简")
            _t2s = False
    return _t2s.convert(text) if _t2s else text


# 维基文库 epub 常见的网页导航/版权噪声，命中即丢弃整行
NOISE_KEYS = (
    "姊妹计划", "图册分类", "数据项", "维基大典", "阅文言", "维基百科", "维基文库",
    "隐私政策", "cookie", "使用条款", "本页面", "此条目", "新条目", "登录",
    "创建帐户", "创建账号", "医疗意见", "医药之指导", "合格之专业",
)
# 古籍篇章标题：脏腑经络先后病脉证第一 / 胸痹心痛短气病脉证治第九 / 辨太阳病脉证并治上
CHAPTER_RE = re.compile(
    r"^(?:辨)?[\u4e00-\u9fa5]{2,18}(?:病脉证(?:并?治)?|脉证(?:并?治)?)(?:第[一二三四五六七八九十百]+|[上中下])$"
)


def _clean_book_lines(raw_lines):
    """清洗网页噪声行，并把识别出的古籍篇名行提升为 ## 标题"""
    out = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if any(k.lower() in s.lower() for k in NOISE_KEYS):
            continue
        if re.fullmatch(r"[←→\-\s|:·•*]+", s):   # 纯导航符号行
            continue
        if re.match(r"^(?:作者|书名|出版社|出版时间|出版日期|译者|校注)[:：]", s):
            continue
        if len(s) <= 30 and CHAPTER_RE.match(s):
            out.append(f"## {s}")
        else:
            out.append(s)
    return out


def read_epub(path):
    """提取 epub 章节标题与正文，转为 '#书名 / ##篇章' 的 Markdown 文本"""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup, NavigableString

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = epub.read_epub(path, options={"ignore_ncx": True})
    meta = book.get_metadata("DC", "title")
    title = meta[0][0] if meta else os.path.splitext(os.path.basename(path))[0]
    parts = [f"# {title}"]
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            h.replace_with(NavigableString(f"\n## {h.get_text(strip=True)}\n"))
        lines = _clean_book_lines(soup.get_text("\n").splitlines())
        body = "\n".join(lines).strip()
        cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
        if cjk < 20 or cjk / max(len(body), 1) < 0.3:   # 跳过封面/英文版权页
            continue
        parts.append(body)
    return "\n\n".join(parts)

def load_documents():
    files = (glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True)
             + glob.glob(os.path.join(DOCS_DIR, "**", "*.txt"), recursive=True)
             + glob.glob(os.path.join(DOCS_DIR, "**", "*.epub"), recursive=True))
    chunks = []
    for fp in files:
        source = os.path.relpath(fp, DOCS_DIR)
        try:
            if fp.endswith(".epub"):
                text = to_simplified(read_epub(fp))
                pieces = split_markdown(text)
            else:
                with open(fp, "r", encoding="utf-8") as f:
                    text = to_simplified(f.read())
                pieces = split_markdown(text) if fp.endswith(".md") else split_plaintext(text, source)
        except Exception as e:
            print(f"      [跳过] {source} 解析失败: {type(e).__name__}: {e}")
            continue
        for content, title_path in pieces:
            if content.strip():
                chunks.append({"text": content, "source": source, "title": title_path})
    return chunks

def main():
    print("[1/4] 加载 embedding 模型（首次运行会从 HF 下载，约 100MB）...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    print("[2/4] 读取并切分文档...")
    chunks = load_documents()
    if not chunks:
        print("未找到任何文档，请把 .md / .txt / .epub 放入 data/docs/ 后重试")
        return
    print(f"      共切出 {len(chunks)} 个文本块")

    print("[3/4] 生成向量...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True
    )

    print("[4/4] 写入 Chroma 向量库...")
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)   # 重建，保证可重复执行
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[f"chunk-{i:04d}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
    )
    print(f"完成！集合 {COLLECTION_NAME}，共 {collection.count()} 块，存储于 {DB_DIR}")

if __name__ == "__main__":
    main()