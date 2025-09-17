from langchain_core.embeddings import Embeddings
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from pathlib import Path

class VectorStore:
    
    def __init__(self, store_type: str, embedding_model: Embeddings, documents: List[Document] = None):
        self.store_type = store_type
        self.embedding_model = embedding_model
        if self.store_type == "faiss":
            if documents:
                self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        else:
            raise Exception(f"Vector store {self.store_type} is not supported")
    
    
    def add_documents(self, documents: List[Document]):
        return self.vectorstore.add_documents(documents=documents)
    
    def load(self, load_dir_path: str, embedding_model: Embeddings):
        dir = Path(load_dir_path)
        if not dir.exists():
            raise FileNotFoundError(f"Directory {dir} not found")
        
        if self.store_type == "faiss":
            self.vectorstore = FAISS.load_local(
                folder_path=load_dir_path,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )

    def save(self, save_dir_path: str):
        dir = Path(save_dir_path)
        if not dir.exists():
            dir.mkdir(parents=True)
        
        if self.store_type == "faiss":
            self.vectorstore.save_local(folder_path=dir)

    