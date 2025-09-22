from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from utils.state import AgentState
from typing import List, Union
from langchain_core.messages import AIMessage, HumanMessage


class HistorySummarizer:

    def __init__(self, llm: BaseChatModel, prompt: str):
        self._llm = llm
        self._prompt = prompt

    def summarize(self, state: AgentState) -> AgentState:
        
        question = state["original_question"]
        history = state["history"]

        prompt = PromptTemplate.from_template(self._prompt).invoke({"question": question, "history": history})

        question_with_history_context = self._llm.invoke(prompt).content.strip()

        state["messages"].append(AIMessage(f"{question} -> {question_with_history_context}"))
        state["original_question"] = question_with_history_context
        state["question"] = question_with_history_context

        return state 