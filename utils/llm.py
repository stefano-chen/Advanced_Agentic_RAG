from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

class LLMModel:
    
    def __init__(self, llm_provider: str, llm_model: str, llm_host: str):
        if llm_provider == "lm-studio":
            self._llm = ChatOpenAI(name=llm_model, base_url=llm_host, api_key="not needed")
        elif llm_provider == "google":
            self._llm = ChatGoogleGenerativeAI(model=llm_model)
        else:
            raise Exception(f"LLM provider {llm_provider} not supported")
        
    def get(self):
        return self._llm