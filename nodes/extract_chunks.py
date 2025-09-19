from utils.state import AgentState

def extract_chunks(state: AgentState) -> AgentState:
    chunks = state["messages"][-1].content.split("\n\n")
    state["chunks"] = chunks
    return state