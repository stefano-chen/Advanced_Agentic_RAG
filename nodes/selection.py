from typing import List, Dict, Any
from utils.state import AgentState
from langchain_core.messages import AIMessage

class ChunckSelection:

    def __init__(self, strategies: List[str], options: Dict[str, Dict[str, Any]]):
        self._strategies = strategies
        self._options = options

    def _selection_by_threshold(self, chunks: List[str], scores: List[float], options: Dict[str, Any]):
        if not options:
            raise Exception("threshold options not found")
        if "min" not in options:
            raise Exception("min field not found in threshold options")
        threshold = options["min"]
        selected_chunks = []
        selected_scores = []
        for i in range(len(chunks)):
            if scores[i] >= threshold:
                selected_chunks.append(chunks[i])
                selected_scores.append(scores[i])
        return selected_chunks, selected_scores

    def _selection_by_topk(self, chunks: List[str], scores: List[float], options: Dict[str, Any]):
        if not options:
            raise Exception("topk options not found")
        if "k" not in options:
            raise Exception("k field not found in topk options")
        k = options["k"]
        # Get sorted indices based on scores
        indices = sorted(range(len(scores)), key=lambda i : scores[i], reverse=True)
        # Reorder chunks using sorted indices
        sorted_chunks = [chunks[i] for i in indices]

        return sorted_chunks[:k]

    def select(self, state: AgentState) -> AgentState:
        chunks = state['chunks']
        scores = state['reranking_score']

        for strategy in self._strategies:
            if strategy == "threshold":
                strategy_option = self._options.get("threshold")
                chunks, scores = self._selection_by_threshold(chunks, scores, strategy_option)
            elif strategy == "topk":
                strategy_option = self._options.get("topk")
                chunks = self._selection_by_topk(chunks, scores, strategy_option)
            else:
                raise NotImplementedError(f"selection strategy {strategy} not supported")
        
        state["messages"].append(AIMessage(f"{len(chunks)} chunks selected"))
        state["chunks"] = chunks
        state["reranking_score"] = None

        return state


