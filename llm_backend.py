#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
llm_backend.py — 统一 LLM 后端（云端优先 + 本地降级）

- 配置了 ZHIPU_API_KEY 时优先调用智谱 GLM-4-Flash（免费、OpenAI 兼容格式）

- 无 key / 超时 / 鉴权失败 / 限流 / 断网时，自动降级本地 Qwen1.5-1.8B(SFT)

- 本地模型懒加载：云端正常时完全不加载，省显存、启动快
"""
import os

import requests
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

# 从项目根目录 .env 读取本地配置（该文件已被 .gitignore 忽略，不会上传）
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

LLM_NAME = "Qwen/Qwen1.5-1.8B"
SFT_ADAPTER = "D:/SFT/models/sft/best"

CLOUD_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
CLOUD_MODEL = "glm-4-flash"   # 智谱长期免费模型
CLOUD_TIMEOUT = 20            # 秒，超时即降级，避免请求长时间挂起

class LLMHub:
    def __init__(self):
        self.api_key = os.getenv("ZHIPU_API_KEY", "").strip()
        self._local_model = None
        self._local_tokenizer = None

    # ---------- 云端 ----------
    def _cloud_generate(self, messages, max_tokens):
        resp = requests.post(
            CLOUD_URL,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": CLOUD_MODEL,
                "messages": messages,
                "temperature": 0.3,        # 低温度：问答要忠实，不要发散
                "max_tokens": max_tokens,
            },
            timeout=CLOUD_TIMEOUT,
        )
        if resp.status_code != 200:
            # 401 key 无效、429 限流/额度用完等，都交给上层降级
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ---------- 本地（懒加载） ----------
    def _ensure_local(self):
        if self._local_model is not None:
            return
        print("[LLM] 加载本地 Qwen1.5-1.8B + SFT adapter（约 10s）...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(LLM_NAME, trust_remote_code=True)
        config._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, config=config, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        if os.path.exists(SFT_ADAPTER):
            model = PeftModel.from_pretrained(model, SFT_ADAPTER)
            model = model.merge_and_unload()
        model.eval()
        self._local_model, self._local_tokenizer = model, tokenizer

    def _local_generate(self, chatml_prompt, max_new_tokens):
        self._ensure_local()
        model, tokenizer = self._local_model, self._local_tokenizer
        inputs = tokenizer(chatml_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    # ---------- 统一入口 ----------
    def generate(self, messages, chatml_prompt, max_new_tokens=320):
        """
        messages:     OpenAI 格式 [{"role":..,"content":..}]，供云端使用
        chatml_prompt: 拼好的 ChatML 字符串，供本地模型使用
        返回 (回答文本, 实际使用的后端名)
        """
        if self.api_key:
            try:
                return self._cloud_generate(messages, max_new_tokens), "cloud(glm-4-flash)"
            except Exception as e:
                print(f"[LLM] 云端不可用（{type(e).__name__}: {e}），自动降级本地模型")
        else:
            print("[LLM] 未配置 ZHIPU_API_KEY，直接使用本地模型")
        return self._local_generate(chatml_prompt, max_new_tokens), "local(qwen1.5-1.8b)"
