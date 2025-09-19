from typing import List, TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    context: str
    chunks: List[str]
    original_question: str
    reranking_score: List[float]

    @classmethod
    def create(cls, messages=[], question=""):
        return AgentState(
            messages=messages, 
            question=question,
            context="",
            chunks=[],
            original_question=question,
            reranking_score=[]
        )