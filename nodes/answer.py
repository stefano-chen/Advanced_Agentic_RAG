
from utils.state import AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage

class GenerateAnswer:

    def __init__(self, llm: BaseChatModel, prompt: str):
        self._prompt = prompt
        self._llm = llm

    def generate_answer(self, state: AgentState) -> AgentState:
        question = state["original_question"]
        context = state["context"]
        prompt = PromptTemplate.from_template(self._prompt).invoke({"question": question, "context": context})

        response = self._llm.invoke(prompt)

        state["messages"].append(response)

        return state
        
        