import gradio as gr
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss
import os

os.environ["SERPAPI_API_KEY"] = "e65622355785aba531fe0f3733c6c429e3ec43457c916a0c3006e6f81d433369"
# 构造 AutoGPT 的工具集
search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

# OpenAI Embedding 模型
embeddings_model = OpenAIEmbeddings()
# OpenAI Embedding 向量维数
embedding_size = 1536
# 使用 Faiss 的 IndexFlatL2 索引
index = faiss.IndexFlatL2(embedding_size)
# 实例化 Faiss 向量数据库
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# 构建autogpt agent
agent = AutoGPT.from_llm_and_tools(
    ai_name="Jarvis",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(), # 实例化 Faiss 的 VectorStoreRetriever
)

def run_autogpt(input):
    return agent.run(input)

iface = gr.Interface(fn=run_autogpt, inputs="text", outputs="text")

def launch_gradio():

    iface = gr.Interface(
        fn=run_autogpt,
        title="AutoGPT",
        inputs=gr.inputs.Textbox(lines=5, label="Your Question"),
        outputs="text"
    )

    iface.launch(share=True, server_name="0.0.0.0")

if __name__ == '__main__':
    launch_gradio()