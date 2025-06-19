# ingest.py
import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/"
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL_NAME = "nomic-embed-text" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def load_documents(path):
    """加载指定路径下的文档 (不使用 loader_map 的备选方案)"""
    all_files = []
    # 支持的扩展名和对应的加载器
    supported_loaders = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    
    print(f"正在从 '{path}' 目录加载文档...")
    # 递归查找所有文件
    for ext in supported_loaders:
        all_files.extend(glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True))
    
    # 去重，以防文件被多次匹配 (例如，如果 glob 模式不够精确)
    unique_files = sorted(list(set(all_files)))
    
    loaded_documents = []
    if not unique_files:
        print(f"在 '{path}' 目录下没有找到支持的文件类型 ({', '.join(supported_loaders.keys())})。")
        return loaded_documents

    print(f"找到 {len(unique_files)} 个文件进行加载: {unique_files}")

    for file_path in unique_files:
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in supported_loaders:
                loader_class = supported_loaders[file_ext]
                print(f"  正在使用 {loader_class.__name__} 加载: {file_path}")
                loader_instance = loader_class(file_path) # 假设加载器都接受 file_path作为第一个参数
                if loader_class == TextLoader:
                    loader_instance = TextLoader(file_path, encoding='utf-8')

                docs_from_file = loader_instance.load()
                if docs_from_file: # 确保加载到了内容
                    # 为每个文档块添加来源元数据
                    for doc in docs_from_file:
                        doc.metadata["source"] = os.path.basename(file_path) # 添加文件名作为source
                    loaded_documents.extend(docs_from_file)
                else:
                    print(f"  警告: {loader_class.__name__} 未能从 {file_path} 加载任何内容。")
            else:
                print(f"  跳过不支持的文件类型: {file_path}") # 理论上不应该发生，因为glob已经筛选
        except Exception as e:
            print(f"  加载文件 {file_path} 时出错: {e}")
            continue # 继续加载其他文件

    if loaded_documents:
        print(f"成功加载并处理了 {len(loaded_documents)} 个文档片段。")
    else:
        print(f"未能从找到的文件中加载任何文档片段。")
        
    return loaded_documents

def split_documents(documents):
    """将文档分割成小块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # 添加块在原文中的起始位置索引，有助于溯源
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文档被分割成了 {len(chunks)} 个小块。")
    return chunks

def initialize_embeddings():
    """初始化Ollama嵌入模型"""
    print(f"正在初始化嵌入模型: {EMBEDDING_MODEL_NAME}...")
    # 直接使用 localhost
    ollama_base_url_corrected = "http://localhost:11434" 

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=ollama_base_url_corrected)
    print(f"使用Ollama服务地址: {ollama_base_url_corrected}")

    # 测试嵌入模型是否工作
    try:
        embeddings.embed_query("测试嵌入模型")
        print("嵌入模型初始化成功。")
    except Exception as e:
        print(f"嵌入模型初始化失败: {e}")
        print(f"请确保Ollama服务正在运行在 {ollama_base_url_corrected}，并且模型 '{EMBEDDING_MODEL_NAME}' 已经通过 `ollama pull {EMBEDDING_MODEL_NAME}` 拉取。")
        raise
    return embeddings

def create_and_persist_vector_store(chunks, embeddings):
    """创建向量数据库并持久化存储"""
    print(f"正在创建向量数据库并将其持久化存储到: {VECTOR_STORE_PATH}...")
    # 直接创建或加载（如果已存在）Chroma数据库
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    print("向量数据库创建并持久化成功！")
    return vector_store

def main():
    print("--- 开始处理知识库 ---")

    # 1. 加载文档
    documents = load_documents(DATA_PATH)
    if not documents:
        print("没有文档可供处理。脚本终止。")
        return

    # 2. 分割文档
    chunks = split_documents(documents)
    if not chunks:
        print("文档分割后没有产生任何文本块。脚本终止。")
        return

    # 3. 初始化嵌入模型
    try:
        embeddings = initialize_embeddings()
    except Exception:
        return # 初始化失败时，错误信息已打印

    # 4. 创建并持久化向量数据库
    create_and_persist_vector_store(chunks, embeddings)

    print("--- 知识库处理完成 ---")

if __name__ == "__main__":
    main()