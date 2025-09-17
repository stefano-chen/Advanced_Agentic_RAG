from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
    
    def __init__(self, file_type: str, file_path: str):
        self.type = file_type
        if self.type.lower() == "pdf":
            self.loader = PyPDFLoader(file_path=file_path)
        else:
            raise Exception(f"File type {file_type} not supported")
        
    def load(self):
        return self.loader.load()