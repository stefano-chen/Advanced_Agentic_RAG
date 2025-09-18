from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Callable, Union, Literal
from langchain_core.tools import BaseTool
from state import AgentState
from langchain.prompts import PromptTemplate

class ToolCalling:

    def __init__(self, llm: BaseChatModel, prompt: str, tools: List[Union[Callable, BaseTool]]):
        self._llm = llm.bind_tools(tools)
        self._prompt = prompt

    def choose(self, state: AgentState) -> AgentState:
        question = state['question']
        context = state['context']
        prompt = PromptTemplate.from_template(self._prompt).invoke({"question": state['question'], "context": state['context']})
        response = self._llm.invoke(prompt)

        state['messages'].append(response)

        return state
    

def tool_condition(state: AgentState) -> Literal["retrieve", "respond"]:
    last_msg = state['messages']

    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "retrieve"
    return "respond"