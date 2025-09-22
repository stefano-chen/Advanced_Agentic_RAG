from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama

class LLMModel:
    
    def __init__(self, config: dict[str, str]):
        if config["llm_provider"] == "lm-studio":
            self._llm = ChatOpenAI(name=config["llm_model"], base_url=config["llm_host"], api_key="not needed", temperature=0)
        elif config["llm_provider"] == "google":
            self._llm = ChatGoogleGenerativeAI(model=config["llm_model"], temperature=0)
        elif config["llm_provider"] == "openai":
            self._llm = ChatOpenAI(name=config["llm_model"], base_url=config["llm_host"], temperature=0)
        elif config["llm_provider"] == "ollama":
            self._llm = ChatOllama(model=config["llm_model"], temperature=0)
        else:
            raise NotImplementedError(f"LLM provider {config["llm_provider"]} not supported")
        
    def get(self):
        return self._llm