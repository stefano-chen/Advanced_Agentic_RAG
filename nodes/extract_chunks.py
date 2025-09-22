from utils.state import AgentState
from langchain_core.messages import AIMessage

def extract_chunks(state: AgentState) -> AgentState:
    chunks = state["messages"][-1].content.split("\n\n")
    state["chunks"] = chunks

    state["messages"].append(AIMessage(f"{len(chunks)} chunks extracted"))

    return state