from typing import List, TypedDict, Annotated, Union
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    context: str
    chunks: Union[List[str], None]
    original_question: str
    reranking_score: Union[List[float], None]

    @classmethod
    def create(cls, messages=[], question=""):
        return AgentState(
            messages=messages, 
            question=question,
            context="",
            chunks=None,
            original_question=question,
            reranking_score=None
        )