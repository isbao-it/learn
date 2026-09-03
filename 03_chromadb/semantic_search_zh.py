"""
阶段3练习（进阶）：用中文向量模型做语义检索
场景：解决默认英文模型对中文效果差的问题
对应项目：任务卡 05 混合检索的 Dense 检索支路（生产选型）
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 1️⃣ 中文 embedding 模型（BAAI 出品，中文语义理解强）
#    首次运行会从 HuggingFace 镜像下载（约100MB），之后永久缓存
ef = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")

client = chromadb.PersistentClient(path="./chroma_data")
col = client.get_or_create_collection("viral_cases_zh", embedding_function=ef)

cases = [
    {"id": "1", "doc": "为什么你越努力越内耗？三个原因拆穿你的情绪黑洞", "title": "内耗视频", "views": 1200000},
    {"id": "2", "doc": "月薪3千和3万的区别，不是努力是选择", "title": "职场搞钱", "views": 3500000},
    {"id": "3", "doc": "每天Emo到崩溃？试试这3个自救方法", "title": "情绪自救", "views": 890000},
    {"id": "4", "doc": "普通人怎么存下第一个100万", "title": "理财赚钱", "views": 2100000},
]

col.add(
    ids=[c["id"] for c in cases],
    documents=[c["doc"] for c in cases],
    metadatas=[{"title": c["title"], "views": c["views"]} for c in cases],
)
print(f"✅ 已用中文模型存入 {col.count()} 条案例")
print()

queries = ["焦虑怎么办", "怎么搞钱", "心态崩了想哭"]

for q in queries:
    print(f"🔍 搜索：「{q}」")
    results = col.query(query_texts=[q], n_results=2)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for i, (d, m, dist) in enumerate(zip(docs, metas, dists), 1):
        print(f"   {i}. [{m['title']}] {d}")
        print(f"      距离 = {dist:.4f}")
    print()
