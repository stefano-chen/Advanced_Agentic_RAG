from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Literal
from langgraph.graph import MessagesState
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from utils.state import AgentState

class QueryValidation:

    def __init__(self, llm: BaseChatModel, prompt: str, topics: List[str]):
        self._llm = llm
        self._prompt = prompt
        self._topics = topics

    def validate(self, state: AgentState) -> AgentState:
        question = state["original_question"]
        prompt_template = PromptTemplate.from_template(self._prompt)
        prompt = prompt_template.invoke({"question": question, "topics": self._topics})
        response = self._llm.invoke(prompt)
        state["messages"].append(AIMessage(f"Is \"{question}\" related with at least one of this topics {self._topics}? {response.content}"))
        return state
        
def is_related(state: AgentState) -> Literal["yes", "no"]:
    last_msg = state["messages"][-1]
    if "yes" in last_msg.content.lower():
        return "yes"
    return "no"