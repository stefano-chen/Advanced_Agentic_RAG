
from utils.state import AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage

class QueryTransform:

    def __init__(self, type: str, kwargs: dict[str, any], llm: BaseChatModel, prompts: dict[str, str]):
        self._llm = llm

        if type == "step-back":
            self._prompt = prompts.get("step-back", None)
            self._max_char = kwargs.get("max_char", 100)
        elif type == "hyde":
            self._prompt = prompts.get("hyde", None)
            self._max_char = kwargs.get("max_char", 500)
        else:
            raise Exception(f"Query transformation {type} is not supported")

    
    def transform(self, state: AgentState) -> AgentState:
        if not self._prompt:
            raise Exception("Query transformation prompt not Found")
        
        question = state["question"]

        prompt = PromptTemplate.from_template(self._prompt).invoke({"max_char": self._max_char, "question": question})

        rewritten_question = self._llm.invoke(prompt).content
        state["messages"].append(AIMessage(f"\"{question}\" -> {rewritten_question}"))
        state["question"] = rewritten_question
        return state
        
