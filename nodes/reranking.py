from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from utils.state import AgentState
from typing import List
from langchain.prompts import PromptTemplate
import numpy as np
from langchain_core.messages import AIMessage

# class Grade(BaseModel):
#     score: float = Field(description="A score between 0 and 1, that measure how much a chunk is related to a user's question")

class Reranking:

    def __init__(self, llm: BaseChatModel, strategies: List[str], weights: List[float], prompts: dict[str, str]):
        self._llm = llm
        self._prompts = prompts
        self._strategies = strategies
        self._weights = weights

    def _calculate_semantic_score(self, question: str, chunks: List[str]):
        scores = []
        prompt_template = self._prompts.get("reranking")
        if not prompt_template:
            raise Exception("Reranking prompt not Found")
        
        for chunk in chunks:
            prompt = PromptTemplate.from_template(prompt_template).invoke({"question": question, "chunk": chunk})
            grade = self._llm.invoke(prompt).content
            try:
                scores.append(float(grade))
            except:
                scores.append(0)
        
        return scores

    def rerank(self, state: AgentState) -> AgentState:
        question = state["original_question"]
        chunks = state["messages"][-1].content.split("\n\n")
        scores_per_strategy = []

        for strategy in self._strategies:
            if strategy == "semantic":
                scores_per_strategy.append(self._calculate_semantic_score(question, chunks))

        matrix = np.array(scores_per_strategy)
        weights = np.array(self._weights)

        if weights.sum() != 1:
            raise Exception("reranking weights must sum to 1")

        final_score = np.average(matrix, axis=0, weights=weights).tolist()
        
        state["messages"].append(AIMessage(f"reranking score: {final_score}"))
        state["reranking_score"] = final_score
        state["chunks"] = chunks
        
        return state