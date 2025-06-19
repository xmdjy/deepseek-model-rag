# qa_app.py
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM 
from langchain_ollama import OllamaEmbeddings # 新的导入方式
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

VECTOR_STORE_PATH = "vector_store"
LLM_MODEL_NAME = "deepseek-r1:7b"
EMBEDDING_MODEL_NAME = "nomic-embed-text" # 确保与 ingest.py 一致

def initialize_llm():
    """初始化Ollama大语言模型"""
    print(f"正在初始化大语言模型: {LLM_MODEL_NAME}...")
    
    # 直接使用 localhost!
    ollama_base_url_corrected = "http://localhost:11434" 
    
    print(f"尝试使用Ollama服务地址: {ollama_base_url_corrected}")
    
    llm = OllamaLLM(model=LLM_MODEL_NAME, base_url=ollama_base_url_corrected)

    try:
        llm.invoke("你好，请确认连接。") 
        print("大语言模型初始化成功。")
    except Exception as e:
        print(f"大语言模型初始化失败: {e}")
        print(f"请确保Ollama服务正在运行在 {ollama_base_url_corrected}，并且模型 '{LLM_MODEL_NAME}' 已经通过 `ollama pull {LLM_MODEL_NAME}` 拉取。")
        raise
    return llm

def initialize_embeddings():
    """初始化Ollama嵌入模型"""
    print(f"正在初始化嵌入模型: {EMBEDDING_MODEL_NAME}...")

    # 直接使用 localhost!
    ollama_base_url_corrected = "http://localhost:11434" 
    
    print(f"尝试使用Ollama嵌入服务地址: {ollama_base_url_corrected}")
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=ollama_base_url_corrected)

    try:
        embeddings.embed_query("测试嵌入模型")
        print("嵌入模型初始化成功。")
    except Exception as e:
        print(f"嵌入模型初始化失败: {e}")
        print(f"请确保Ollama服务正在运行在 {ollama_base_url_corrected}，并且模型 '{EMBEDDING_MODEL_NAME}' 已经通过 `ollama pull {EMBEDDING_MODEL_NAME}` 拉取。")
        raise
    return embeddings

def load_vector_store(embeddings_func): # 修改为接受 embeddings_func
    """加载向量数据库"""
    # ... (内部逻辑不变, 但确保它使用的是传入的 embeddings_func)
    if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
        print(f"错误：向量数据库目录 '{VECTOR_STORE_PATH}' 不存在或为空。")
        print("请先运行 `ingest.py` 来创建和填充向量数据库。")
        return None

    print(f"正在从 '{VECTOR_STORE_PATH}' 加载向量数据库...")
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings_func # 使用传入的 embeddings 对象
    )
    print("向量数据库加载成功。")
    return vector_store

def create_qa_chain(llm, vector_store):  
    """创建问答链"""
    prompt_template = """请根据下面提供的上下文来回答问题。答案应尽可能简洁，并直接来源于上下文。
如果上下文中没有足够的信息来回答问题，请明确说明你不知道，不要试图编造答案。

上下文:
{context}

问题: {question}

答案:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("问答链创建成功。")
    return qa_chain

def main():
    print("--- 启动个人知识库问答应用 ---")

    try:
        llm = initialize_llm()
        embeddings = initialize_embeddings() # 这个 embeddings 对象需要传递给 vector_store
    except Exception:
        print("初始化失败，应用无法启动。")
        return

    vector_store = load_vector_store(embeddings) # 传递 embeddings 对象
    if vector_store is None:
        return

    qa_chain = create_qa_chain(llm, vector_store)

    print("\n可以开始提问了！输入 '退出' 来结束程序。")
    while True:
        query = input("\n你问: ")
        if query.lower() == '退出':
            print("感谢使用，再见！")
            break
        if not query.strip():
            continue

        print("正在思考...")
        try:
            result = qa_chain.invoke({"query": query}) # 对于新版 RetrievalQA，可能只需要 query
            answer = result.get("result", "抱歉，未能生成答案。")
            source_documents = result.get("source_documents")

            print("\n模型回答:")
            print(answer)

            if source_documents:
                print("\n参考来源:")
                for i, doc in enumerate(source_documents):
                    source_info = f"  - 来源 {i+1}: "
                    if 'source' in doc.metadata:
                        source_info += f"文件 '{os.path.basename(doc.metadata['source'])}'"
                    if 'start_index' in doc.metadata:
                         source_info += f" (起始位置: {doc.metadata['start_index']})"
                    print(source_info)
        except Exception as e:
            print(f"处理请求时发生错误: {e}")

if __name__ == "__main__":
    main()