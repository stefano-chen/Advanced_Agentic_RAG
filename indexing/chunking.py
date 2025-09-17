from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunking:
    def __init__(self, chunking_strategy: str, chunking_options: Dict[str, Any]):
        self.chunking_strategy = chunking_strategy

        if self.chunking_strategy.lower() == "window":
            self.text_splitter = RecursiveCharacterTextSplitter(**chunking_options)
        else:
            raise Exception(f"Chunking strategy {self.chunking_strategy} not supported")
        
    def apply(self, documents: List[Document]):
        return self.text_splitter.split_documents(documents=documents)
