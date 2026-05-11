import os
import glob
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import OllamaEmbeddings

def get_or_update_vectorstore():
    
    current_files = set(glob.glob("./data/**/*.pdf", recursive=True))
    local_embeddings = OllamaEmbeddings(model="bge-m3")
    
    SEPERATORS = ['\n#{1,6}', '```\n', '\n\\*\\*\\*+\n', '\n---+\n', '\n___+\n', '\n', ' ', '']
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=SEPERATORS)

    vector_db_path = "faiss_index"
    state_file = os.path.join(vector_db_path, "indexed_files.json")

    if os.path.exists(vector_db_path):
        vectorstore = FAISS.load_local(
            folder_path=vector_db_path, 
            embeddings=local_embeddings, 
            allow_dangerous_deserialization=True 
        )
        
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                indexed_files = set(json.load(f))
        else:
            indexed_files = set()
            
        new_files = current_files - indexed_files
        
        if new_files:
            new_docs = []
            for file_path in new_files:
                new_docs.extend(PyPDFLoader(file_path).load())
                
            if new_docs:
                new_splits = text_splitter.split_documents(new_docs)
                vectorstore.add_documents(new_splits)   
                vectorstore.save_local(vector_db_path)  

            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(list(current_files), f)
    else:
        loader = DirectoryLoader(
            path='./data',
            glob='**/*.pdf',
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=local_embeddings, 
            distance_strategy=DistanceStrategy.COSINE
        )
        vectorstore.save_local(vector_db_path) 
        
        os.makedirs(vector_db_path, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(list(current_files), f)
            
    return vectorstore