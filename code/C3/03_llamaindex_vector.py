import faiss
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 改为用Faiss作为向量存储后端
d = 512
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=Settings.embed_model)

# 3. 创建索引并持久化到本地
# index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")
