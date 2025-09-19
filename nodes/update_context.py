from utils.state import AgentState
from langchain_core.messages import AIMessage

def update_context(state: AgentState):

    new_context = ""

    message = None

    if state["chunks"]:
        new_context = "\n\n".join(state['chunks'])
        message = AIMessage("Context Updated")
        state["chunks"] = []
        state["reranking_score"] = []
        state['context'] = state['context'] + new_context
    else:
        message = AIMessage("No new Context")

    state['messages'].append(message)

    return state