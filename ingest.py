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

def load_documents():
    files = (glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True)
             + glob.glob(os.path.join(DOCS_DIR, "**", "*.txt"), recursive=True))
    chunks = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.relpath(fp, DOCS_DIR)
        pieces = split_markdown(text) if fp.endswith(".md") else split_plaintext(text, source)
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
        print("未找到任何文档，请把 .md / .txt 放入 data/docs/ 后重试")
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