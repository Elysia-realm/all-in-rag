import faiss
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法律顾问",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
documents = [Document(text=text) for text in texts]

# print(len(Settings.embed_model.get_text_embedding("你好世界")))
# 嵌入维度, BAAI/bge-small-zh-v1.5的嵌入维度为512
d = 512
faiss_index = faiss.IndexFlatL2(d)  # 使用L2距离的平坦索引

# 3. 创建Faiss向量存储适配器
vector_store = FaissVectorStore(faiss_index)
# 创建存储上下文，将Faiss作为向量存储后端
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# (LlamaIndex 会自动完成：分块 → 生成嵌入 → 存入 FAISS)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=Settings.embed_model)

# 上面我们是hard-code生成文本并创建Faiss索引，但也可以直接从磁盘中加载
# 而要加载Faiss向量索引，那就只能加载用Faiss存的向量索引
# 03中一开始是用llamaindex默认的向量存储，因此将其改为Faiss
vector_store = FaissVectorStore.from_persist_dir("./llamaindex_index_store")
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./llamaindex_index_store")
index = load_index_from_storage(storage_context=storage_context)

# 4. 构建查询模块
# query_engine = index.as_query_engine() # 使用查询引擎的方式必须配合LLM
retriever = index.as_retriever(similarity_top_k=1)

# 5. 用户查询
# response = query_engine.query("llamaIndex是什么？")
responses = retriever.retrieve("张三是谁？")
print("共查询到 {} 条相关内容。".format(len(responses)))
print(responses[0].text)

