from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Literal
from state import AgentState
from langchain.prompts import PromptTemplate

class QueryValidation:

    def __init__(self, llm: BaseChatModel, prompt: str, topics: List[str]):
        self._llm = llm
        self._prompt = prompt
        self._topics = topics

    def validate(self, state: AgentState) -> Literal["yes", "no"]:
        prompt_template = PromptTemplate.from_template(self._prompt + "/no_think")
        prompt = prompt_template.invoke({"question": state["question"], "topics": self._topics})
        response = self._llm.invoke(prompt)
        print(response)
        if "yes" in response.content.lower():
            return 'yes'
        return 'no'
        