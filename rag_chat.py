#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_chat.py — RAG 问答链路（命令行验证版）
流程：问题 → bge 向量化 → Chroma 检索 top-k → 拼 Prompt → 本地 Qwen(SFT) 生成
用法：python rag_chat.py "什么是治未病？"
"""
import os
import re
import sys

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from llm_backend import LLMHub

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ROOT, "vector_store")
COLLECTION_NAME = "tcm_docs"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

TOP_K = 3
SCORE_THRESHOLD = 0.55   # 相似度阈值，低于此值视为噪声不送入 Prompt
MAX_NEW_TOKENS = 320

DISCLAIMER = "\n\n⚠️ 本回答基于中医典籍整理，仅供学习参考，不构成诊疗建议，身体不适请及时就医。"


def retrieve(collection, embedder, question, top_k=TOP_K):
    """检索 top-k 相关块并按阈值过滤，返回 (文本, 来源标题, 相似度) 列表"""
    q_vec = embedder.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=top_k)
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    hits = [(d, m.get("title", m.get("source", "")), 1 - dist)
            for d, m, dist in zip(docs, metas, dists)]
    return [h for h in hits if h[2] >= SCORE_THRESHOLD] or hits[:1]  # 全低时保底保留最相似的一条


INSTRUCTION = (
    "你是中医典籍问答助手。请只依据下面【参考资料】回答问题，"
    "用简洁中文分点陈述，必要时可引用原文；"
    "资料中没有依据的内容一律不要编造，没有相关资料就回答'典籍中未提及'；"
    "回答中禁止出现'场景''注：''版本差异'等与资料无关的标记，"
    "也不要使用代码块（```）或表格，用普通中文和序号表达。"
)


def build_inputs(question, contexts):
    """同时产出云端 messages 与本地 ChatML（本地模型 SFT 时无 system 角色）"""
    refs = "\n\n".join(f"【资料{i+1}】{c}" for i, (c, _, _) in enumerate(contexts))
    user_content = f"【参考资料】\n{refs}\n\n【问题】{question}"
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
    chatml_prompt = (
        f"<|im_start|>user\n{INSTRUCTION}\n\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return messages, chatml_prompt

def clean_answer(text):
    """清理 SFT 模板残留：截断尾注，去除无效/孤立代码围栏"""
    text = re.split(r"\n?\s*(?:注[：:]|（?场景\d+）?)", text, maxsplit=1)[0]
    # 去掉代码围栏开标记（含模型生成的非法语言名如 ```文本）与孤立闭标记
    text = re.sub(r"```[\u4e00-\u9fa5A-Za-z]*\n?", "", text)
    return text.strip()


def main():
    if len(sys.argv) < 2:
        print('用法: python rag_chat.py "你的问题"')
        return
    question = sys.argv[1]

    print("[1/3] 加载 embedding 模型与生成后端...")
    embedder = SentenceTransformer(
        EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    hub = LLMHub()

    print(f"[2/3] 检索知识库（top-{TOP_K}）...")
    contexts = retrieve(collection, embedder, question)
    for i, (_, title, score) in enumerate(contexts):
        print(f"      命中{i+1} [相似度 {score:.3f}] {title}")

    print("[3/3] 生成回答...\n")
    messages, chatml_prompt = build_inputs(question, contexts)
    answer, backend = hub.generate(messages, chatml_prompt, MAX_NEW_TOKENS)
    answer = clean_answer(answer)
    print("=" * 60)
    print(f"[后端: {backend}]")
    print(answer + DISCLAIMER)
    print("=" * 60)

if __name__ == "__main__":
    main()