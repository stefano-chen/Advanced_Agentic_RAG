from utils.state import AgentState
from langchain_core.messages import AIMessage

def update_context(state: AgentState):

    state['messages'].append(AIMessage("Context Updated"))
    state['context'] = "\n\n".join(state['chunks'])

    return state