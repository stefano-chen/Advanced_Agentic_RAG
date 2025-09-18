
from utils.state import AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate

class GenerateAnswer:

    def __init__(self, llm: BaseChatModel, prompt: dict[str, str]):
        self._prompt = prompt.get("output", None)
        self._llm = llm

    def generate_answer(self, state: AgentState) -> AgentState:
        if not self._prompt:
            raise Exception("Generate Answer prompt not Found")
        question = state["original_question"]
        context = state["context"]
        prompt = PromptTemplate.from_template(self._prompt).invoke({"question": question, "context": context})

        response = self._llm.invoke(prompt)

        state["messages"].append(response)

        return state
        
        