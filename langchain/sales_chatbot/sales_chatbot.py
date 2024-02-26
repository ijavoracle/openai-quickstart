import gradio as gr
import os
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 加载定义在.env文件中的openai_api_Key配置信息
load_dotenv()


def initialize_sales_bot(vector_store_dir: str):
    api_base_url = os.environ["API_BASE_URL"]
    api_key = os.environ["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(
        openai_api_base=api_base_url,
        openai_api_key=api_key
    )
    db = FAISS.load_local(vector_store_dir, embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,  openai_api_base=api_base_url, openai_api_key=api_key)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    scenario = os.environ["scenario"]
    title = "房产销售"
    if scenario == 'car_sales':
        title = "汽车销售"

    demo = gr.ChatInterface(
        fn=sales_chat,
        title=title,
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 根据环境变量确定要初始化的向量数据库  real_estate
    vector_store = "real_estates_sale"
    if os.environ["scenario"] == 'car_sales':
        vector_store = "bmw_sales_data"

    # 初始化销售机器人
    initialize_sales_bot(vector_store)
    # 启动 Gradio 服务
    launch_gradio()
