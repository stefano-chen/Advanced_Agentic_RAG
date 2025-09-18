from utils.state import AgentState
from langchain_core.messages import AIMessage

def update_context(state: AgentState):

    last_msg = state['messages'][-1]
    state['messages'].append(AIMessage("Context Updated"))
    state['context'] = last_msg.content

    return state