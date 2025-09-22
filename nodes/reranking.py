from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from utils.state import AgentState
from typing import List
from langchain.prompts import PromptTemplate
import numpy as np
from langchain_core.messages import AIMessage
from utils.llm import LLMModel
from utils.embedding import EmbeddingModel

# class Grade(BaseModel):
#     score: float = Field(description="A score between 0 and 1, that measure how much a chunk is related to a user's question")

class Reranking:

    def __init__(self, strategies: List[str], weights: List[float], options: dict[str, dict[str, str]], prompts: dict[str, str]):
        self._options = options
        self._prompts = prompts
        self._strategies = strategies
        self._weights = weights

    def _calculate_semantic_score(self, question: str, chunks: List[str]):
        scores = []
        prompt_template = self._prompts.get("reranking")
        if not prompt_template:
            raise Exception("Reranking prompt not Found")
        
        llm_config = self._options.get("semantic")

        if not llm_config:
            raise Exception("Semantic llm options not Found")

        llm = LLMModel(llm_config).get()

        for chunk in chunks:
            prompt = PromptTemplate.from_template(prompt_template).invoke({"question": question, "chunk": chunk})
            grade = llm.invoke(prompt).content
            try:
                scores.append(float(grade))
            except:
                scores.append(0)
        
        return scores
    
    def _calculate_distance_score(self, question: str, chunks: List[str]):
        embedding_config = self._options.get("distance")

        if not embedding_config:
            raise Exception("Distance embedding options not Found")

        embedding_model = EmbeddingModel(embedding_config).get()

        question_emb = np.array(embedding_model.embed_query(question))
        scores = []
        for chunk in chunks:
            chunk_emb = np.array(embedding_model.embed_query(chunk))
            distance = np.linalg.norm(question_emb - chunk_emb)
            scores.append((1 / (1 + distance)))
            
        return scores

    def rerank(self, state: AgentState) -> AgentState:
        question = state["original_question"]
        scores_per_strategy = []
        chunks = state["chunks"]
        for strategy in self._strategies:
            if strategy == "semantic":
                scores_per_strategy.append(self._calculate_semantic_score(question, chunks))
            elif strategy == "distance":
                scores_per_strategy.append(self._calculate_distance_score(question, chunks))
            else:
                raise NotImplementedError(f"reranking strategy {strategy} not supported")

        matrix = np.array(scores_per_strategy)
        weights = np.array(self._weights)

        if weights.sum() != 1:
            raise Exception("reranking weights must sum to 1")

        final_score = np.average(matrix, axis=0, weights=weights).tolist()
        
        state["messages"].append(AIMessage(f"weighted average score between all reranking techniques: {final_score}"))
        state["reranking_score"] = final_score

        return state