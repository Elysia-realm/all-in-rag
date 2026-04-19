import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 导入必要的库, 并加载环境变量. 因为API_KEY存在环境变量中
load_dotenv()

# 使用 AIHubmix
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 加载嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载外部知识库
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 通过嵌入模型将外部知识库转化为向量，并构建向量索引
index = VectorStoreIndex.from_documents(docs)

# 构建查询模块
query_engine = index.as_query_engine()

# llamaIndex有默认的提示词模板
print(query_engine.get_prompts())

# 用户查询，隐含了大量过程
# 包括对question通过相似度在向量存储中搜索上下文content，然后将content和question输入llm返回答案
print(query_engine.query("文中举了哪些例子?"))