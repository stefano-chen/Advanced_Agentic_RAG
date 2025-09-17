import json
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from indexing.document_loader import DocumentLoader
from indexing.chunking import Chunking
from indexing.vectorstore import VectorStore
from pathlib import Path
    

if __name__ == "__main__":
    
    with open("./config/populate_config.json", "r") as f:
        config = json.load(f)
    
    if config["embedding_provider"] == "huggingface":
        embedding_model = HuggingFaceEmbeddings(model_name=config["embedding_model"])
    
    data_dir_path = config["data_dir_path"]
    if not os.path.exists(data_dir_path):
        raise NotADirectoryError(f"{data_dir_path} is not a directory")
    
    for entry in os.scandir(data_dir_path):
        if entry.is_file():
            print(f"loading from {entry.path}")
            pages = DocumentLoader(file_type=entry.path.split(".")[-1], file_path=entry.path).load()
            print(f"{len(pages)} pages loaded")
            print(f"Apply chunking strategy {config["chunking_strategy"]}")
            chunks = Chunking(config["chunking_strategy"], config["chunking_options"], embedding_model).apply(pages)
            print(f"{len(chunks)} chunks extracted")
            print(f"Creating Vector store {config["vector_store_type"]}")
            vector_store = VectorStore(config["vector_store_type"], embedding_model, chunks)
            save_dir = Path(config["save_dir_path"]).joinpath(entry.name.split(".")[0])
            print(f"saving vector store at {save_dir}\n")
            vector_store.save(save_dir)
