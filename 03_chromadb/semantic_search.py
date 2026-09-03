"""
阶段3练习：ChromaDB 向量检索
场景：爆款案例语义检索——搜"意思"而不是搜"字面"
对应项目：任务卡 05 三重混合检索的 Dense（语义）检索支路
"""

import chromadb

# 1️⃣ 创建持久化客户端（序列化数据存本地 chroma_data 目录）
client = chromadb.PersistentClient(path="./chroma_data")

# 2️⃣ 获取/创建集合（≈ 数据库的一张表 集合类似关系型数据库中的“表”）
col = client.get_or_create_collection("viral_cases")

# 3️⃣ 写入爆款案例（注意：文案里故意【不写】"焦虑"二字，但语义是焦虑）
cases = [
    {"id": "1", "doc": "为什么你越努力越内耗？三个原因拆穿你的情绪黑洞", "title": "内耗视频", "views": 1200000},
    {"id": "2", "doc": "月薪3千和3万的区别，不是努力是选择", "title": "职场搞钱", "views": 3500000},
    {"id": "3", "doc": "每天Emo到崩溃？试试这3个自救方法", "title": "情绪自救", "views": 890000},
    {"id": "4", "doc": "普通人怎么存下第一个100万", "title": "理财赚钱", "views": 2100000},
]
# Embedding（向量化）：调用默认的文本嵌入模型（通常是 all-MiniLM-L6-v2 这样的轻量级 Sentence Transformer 模型），将每条 doc 文本转换成一个高维浮点数向量（如 384 维）。这个向量在多维空间中代表了文本的语义坐标。
# 存储：将生成的向量与原始文本、元数据一起存入集合。
# add() 是 upsert：id 相同就覆盖，所以重复运行不会越存越多
# 向集合中添加数据
col.add(
    # 使用列表推导式从cases中提取所有案例的id作为文档ID
    ids=[c["id"] for c in cases],
    # 使用列表推导式从cases中提取所有案例的文档内容
    documents=[c["doc"] for c in cases],
    # 使用列表推导式从cases中提取标题和浏览量作为元数据
    metadatas=[{"title": c["title"], "views": c["views"]} for c in cases],
)
# 打印已存储的案例总数，使用✅表情符号表示成功
print(f"✅ 已存入 {col.count()} 条爆款案例")
# 打印空行以增加输出可读性
print()

# 4️⃣ 语义检索：故意用"字面不同、意思相同"的词去搜
queries = ["焦虑怎么办", "怎么搞钱", "心态崩了想哭"]

# 遍历所有查询语句
for q in queries:
    # 打印搜索提示和当前查询内容
    print(f"🔍 搜索：「{q}」")
    # 执行查询，获取最相关的2个结果
    results = col.query(query_texts=[q], n_results=2)
# 当传入 query_texts 时，ChromaDB 会先用同一个 Embedding 模型将查询语句（如“焦虑怎么办”）转化为向量。
# 然后在数据库中计算该查询向量与所有存储向量之间的距离（默认使用余弦距离 Cosine Distance 或 L2 距离）。
# 返回距离最小（即语义最相似）的 Top n_results 条结果。
    # 提取查询结果中的文档内容
    docs = results["documents"][0]
    # 提取查询0结果中的元数据（如标题等）
    metas = results["metadatas"][0]
    # 提取查询结果中的相似度距离
    dists = results["distances"][0]
    # 遍历每个查询结果，并编号输出
    for i, (d, m, dist) in enumerate(zip(docs, metas, dists), 1):
        # 打印结果序号、标题和内容
        print(f"   {i}. [{m['title']}] {d}")
        # 打印相似度距离，保留4位小数，并说明距离越小表示越相似
        print(f"      距离 = {dist:.4f}（越小越像）")
    # 打印空行，分隔不同查询的结果
    print()

# 5️⃣ 对比：传统"字面搜索"的局限
print("=" * 50)
print("📌 对比实验：如果你用传统关键词搜索「焦虑」")
print("   只会命中含「焦虑」二字的文案 —— 而案例库 4 条里「焦虑」出现 0 次")
print("   → 传统搜索返回 0 条，向量搜索却能返回「内耗」「Emo」这些同义内容 ✅")
