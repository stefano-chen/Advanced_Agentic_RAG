from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

class LLMModel:
    
    def __init__(self, config: dict[str, str]):
        if config["llm_provider"] == "lm-studio":
            self._llm = ChatOpenAI(name=config["llm_model"], base_url=config["llm_host"], api_key="not needed")
        elif config["llm_provider"] == "google":
            self._llm = ChatGoogleGenerativeAI(model=config["llm_model"])
        else:
            raise Exception(f"LLM provider {config["llm_provider"]} not supported")
        
    def get(self):
        return self._llm