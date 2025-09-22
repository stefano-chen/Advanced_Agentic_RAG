from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Callable, Union, Literal
from langchain_core.tools import BaseTool
from langchain.prompts import PromptTemplate
from utils.state import AgentState
from langchain_core.messages import AnyMessage, AIMessage

class Choice:
    def __init__(self, llm: BaseChatModel, prompt: str):
        self._llm = llm
        self._prompt = prompt

    def _get_past_tool_calls(self, messages: List[AnyMessage]) -> str:
        past_tool_calls = ""
        for msg in messages:
            if msg.type == "ai" and hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                past_tool_calls += str(msg.tool_calls) + "\n"
        return past_tool_calls
    
    def choose(self, state: AgentState) -> AgentState:
        question = state["original_question"]
        context = state["context"]
        past_tool_calls = self._get_past_tool_calls(state["messages"])
        prompt = PromptTemplate.from_template(self._prompt).invoke({"question": question, "context": context, "past_tool_calls": past_tool_calls})
        response = self._llm.invoke(prompt)
        state["messages"].append(response)
        return state

class ToolCalling:

    def __init__(self, llm: BaseChatModel, prompt: str, tools: List[Union[Callable, BaseTool]]):
        self._llm = llm.bind_tools(tools)
        self._prompt = prompt

    def call(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]

        if "retrieve" in last_msg.content.lower():
            prompt = PromptTemplate.from_template(self._prompt).invoke({"query": state["question"]})
            response = self._llm.invoke(prompt)
            state["messages"].append(response)
        else: 
            state["messages"].append(AIMessage("No tools to call, routing to the generate answer node"))
        return state


def tool_condition(state: AgentState) -> Literal["retrieve", "respond"]:
    last_msg = state['messages'][-1]

    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "retrieve"
    return "respond"