# 中医典籍 RAG 问答助手

基于检索增强生成（RAG）的中医典籍问答系统：**本地 bge 向量检索 + 云端 GLM-4-Flash 优先 / 本地 Qwen1.5-1.8B(SFT) 自动降级**，单卡 RTX 3060 可完整复现。无需框架，RAG 链路全部手写。

## 架构

```
用户问题
   │
   ├─ ① Embedding：BAAI/bge-small-zh-v1.5（本地，512 维）
   ├─ ② 向量检索：Chroma，cosine top-3，相似度阈值 0.55 过滤噪声
   └─ ③ 生成（双后端容错）
         ├─ 优先：智谱 GLM-4-Flash（云端，免费）
         └─ 失败自动降级：本地 Qwen1.5-1.8B + 自训 SFT LoRA
   ↓
回答 + 实际后端标注 + 引用来源（章节、相似度、耗时）
````

## 工程亮点

1. **双后端容错架构**：覆盖无 key / 超时(20s) / 401 鉴权失败 / 429 限流 / 断网五类故障，云端异常时自动降级本地模型，服务始终可用。
2. **本地模型懒加载**：云端成功时不加载 1.8B 模型，省显存、启动快。
3. **结构化分块**：按 Markdown 标题切块并附带"章节路径"，超长块 450 字 + 50 字重叠滑窗，优于定长硬切。
4. **Prompt 贴合训练分布**：本地 SFT 模型训练时无 system 角色，云端走标准 system+user、本地走纯 user ChatML，同一套资料按模型分别构造输入。
5. **生成端清洗**：针对 SFT 模板语料的尾注泄漏（"注：…（场景N）"）做 Prompt 约束 + 正则后处理双重治理。
6. **可解释检索**：每条回答返回引用章节与相似度分数，检索质量可审计。
7. **知识库零代码扩充**：把 .md/.txt 放入 `data/docs/` 重跑 `ingest.py` 即可。

## 快速开始

```powershell
# 1. 环境（Python 3.12）
python -m venv d:\rag_env
d:\rag_env\Scripts\Activate.ps1
pip install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
pip install -r requirements.txt

# 2. 知识库入库（首次需下载 bge 模型，约 100MB）
$env:HF_ENDPOINT="https://hf-mirror.com"
python ingest.py

# 3. 配置云端 key（可选；不配则纯本地运行）
$env:ZHIPU_API_KEY="你的id.secret"     # https://open.bigmodel.cn/usercenter/apikeys

# 4. 启动服务
python app.py
```

| 地址 | 用途 |
|------|------|
| http://localhost:8000 | 网页问答 Demo（显示后端、引用来源） |
| http://localhost:8000/docs | Swagger 接口文档，可在线调试 |
| http://localhost:8000/health | 健康检查（知识库块数、云端是否配置） |

命令行单问：

```powershell
python rag_chat.py "太阳中风和太阳伤寒怎么区分？"
```

## 效果对比（同一问题、同一检索结果）

| 后端 | 回答表现 |
|------|---------|
| GLM-4-Flash（云端） | 准确分点：有汗脉缓 vs 无汗脉紧，忠实于资料 |
| Qwen1.5-1.8B（本地 SFT） | 要点基本正确，但会脑补（如"化验结果"）、表格行列错位 |

**结论**：检索质量达标（命中章节相似度 0.79）时，瓶颈在生成模型容量；双后端设计既保证可复现（纯本地离线），又保证演示质量（云端增强）。

## 目录结构

````
tcm-rag/
├── data/docs/            # 知识库原始文档（中医典籍种子.md）
├── vector_store/         # Chroma 持久化（gitignore，ingest 可重建）
├── ingest.py             # 文档切分 → embedding → 入库
├── rag_chat.py           # 检索 + Prompt + 清洗（命令行入口）
├── llm_backend.py        # 云端/本地双后端 + 自动降级
├── app.py                # FastAPI 服务 + 网页 Demo
├── requirements.txt
├── .env.example
└── .gitignore
````

## 已知局限

- 种子知识库仅 15 块（中医基础理论 + 内经/伤寒论精选），覆盖面有限，可自行放入公版典籍全文扩充。
- 本地 1.8B 模型存在事实性脑补与格式问题，生产场景应使用更大模型或云端 API。
- 目前只支持 .md/.txt，PDF 需先转换或额外引入解析器。

## 合规声明

知识库内容取自公共版权古籍（《黄帝内经》《伤寒论》等）。本系统仅供学习交流，**所有回答不构成诊疗建议，身体不适请及时就医**。请勿上传侵权或付费资料。