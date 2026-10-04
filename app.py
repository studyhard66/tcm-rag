#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — 中医典籍 RAG 问答服务（FastAPI）
启动：python app.py  →  http://localhost:8000
"""
import os
import time
from contextlib import asynccontextmanager

import torch
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from rag_chat import (
    retrieve, build_inputs, clean_answer,
    DB_DIR, COLLECTION_NAME, EMBED_MODEL, TOP_K, MAX_NEW_TOKENS, DISCLAIMER,
)
from llm_backend import LLMHub

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时只加载一次；本地 LLM 在 LLMHub 内懒加载（云端正常时不占显存）
    print("[启动] 加载 embedding 模型与知识库...")
    state["embedder"] = SentenceTransformer(
        EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    state["collection"] = client.get_collection(COLLECTION_NAME)
    state["hub"] = LLMHub()
    print("[启动] 完成，访问 http://localhost:8000")
    yield

app = FastAPI(title="中医典籍 RAG 问答助手", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200)
    max_new_tokens: int = Field(default=320, ge=64, le=800)

@app.get("/health")
def health():
    return {"status": "ok", "chunks": state["collection"].count(),
            "cloud_configured": bool(state["hub"].api_key)}

@app.post("/api/chat")
def chat(req: ChatRequest):
    t0 = time.time()
    contexts = retrieve(state["collection"], state["embedder"], req.question)
    messages, chatml_prompt = build_inputs(req.question, contexts)
    answer, backend = state["hub"].generate(
        messages, chatml_prompt, req.max_new_tokens
    )
    return {
        "answer": clean_answer(answer) + DISCLAIMER,
        "backend": backend,
        "latency_s": round(time.time() - t0, 2),
        "sources": [{"title": t, "score": round(s, 3)} for _, t, s in contexts],
    }

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>中医典籍问答助手</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei",sans-serif;background:#f4f1ea;color:#3a3226;
display:flex;justify-content:center;height:100vh}
.box{width:760px;max-width:100%;display:flex;flex-direction:column;background:#fffdf7;
box-shadow:0 0 24px rgba(120,90,40,.12)}
header{background:#5d7a52;color:#fff;padding:18px 24px}
header h1{font-size:20px;font-weight:600}
header p{font-size:12px;opacity:.85;margin-top:4px}
#chat{flex:1;overflow-y:auto;padding:20px 24px}
.msg{margin:12px 0;line-height:1.7;font-size:15px}
.q{text-align:right}.q span{display:inline-block;background:#5d7a52;color:#fff;
padding:8px 14px;border-radius:14px 14px 4px 14px;max-width:80%;text-align:left;white-space:pre-wrap}
.a{background:#f0ece0;border-radius:4px 14px 14px 14px;padding:12px 16px;white-space:pre-wrap}
.tag{display:inline-block;font-size:11px;padding:1px 8px;border-radius:10px;margin-bottom:6px}
.cloud{background:#dcead3;color:#3d6b2e}.local{background:#f3e2c7;color:#8a5a1e}
.src{font-size:12px;color:#8a7d63;margin-top:8px;border-top:1px dashed #d8cfb8;padding-top:6px}
.dis{font-size:12px;color:#b05a4a;margin-top:8px}
#bar{display:flex;gap:10px;padding:14px 24px;border-top:1px solid #e4dcc8;background:#faf7ee}
textarea{flex:1;border:1px solid #d3c9ae;border-radius:8px;padding:10px;font-size:14px;
resize:none;height:46px;font-family:inherit}
button{background:#5d7a52;color:#fff;border:none;border-radius:8px;padding:0 22px;
font-size:15px;cursor:pointer}button:disabled{opacity:.5}
</style></head><body>
<div class="box">
<header><h1>中医典籍问答助手</h1>
<p>RAG 检索增强 · bge-small-zh + Chroma · GLM-4-Flash 云端优先 / Qwen1.5 本地降级</p></header>
<div id="chat"><div class="msg a">你好，可以问我中医典籍相关问题，例如：桂枝汤的组成、什么是治未病、太阳中风与伤寒的区别。</div></div>
<div id="bar"><textarea id="q" placeholder="输入问题，Enter 发送，Shift+Enter 换行"></textarea>
<button id="send">发送</button></div></div>
<script>
const chat=document.getElementById('chat'),btn=document.getElementById('send'),q=document.getElementById('q');
function add(cls,html){const d=document.createElement('div');d.className='msg '+cls;
d.innerHTML=html;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
async function send(){
  const text=q.value.trim();if(!text)return;
  q.value='';btn.disabled=true;
  add('q','<span>'+text.replace(/</g,'&lt;')+'</span>');
  add('a','<i>思考中…</i>');const node=chat.lastChild;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:text})});
    const d=await r.json();
    const isCloud=d.backend.startsWith('cloud');
    node.innerHTML='<span class="tag '+(isCloud?'cloud':'local')+'">'+d.backend+'</span>'
      +'<div>'+d.answer.replace(/</g,'&lt;')+'</div>'
      +'<div class="src">引用：'+d.sources.map(s=>s.title+' ('+s.score+')').join('；')
      +'　耗时 '+d.latency_s+'s</div>';
  }catch(e){node.textContent='服务异常：'+e;}
  btn.disabled=false;chat.scrollTop=chat.scrollHeight;
}
btn.onclick=send;
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
</script></body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)