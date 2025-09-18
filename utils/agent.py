from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from nodes.query_validation import QueryValidation, is_related
from nodes.tool_calling import ToolCalling, tool_condition
from utils.state import AgentState

def build_agent(llm, prompts, tools, topics):
    graph = StateGraph(AgentState)
    # Insert Graph Node
    graph.add_node("validate_input", QueryValidation(llm, prompts["input_check"], topics).validate)
    graph.add_node("retrieve_or_respond", ToolCalling(llm, prompts["retrieve_respond"], tools).choose)
    graph.add_node("tool_execution", ToolNode(tools))
    # Create Edeges
    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges(
        "validate_input",
        is_related,
        {
            "yes": "retrieve_or_respond",
            "no": END
        }
    )
    graph.add_conditional_edges(
        "retrieve_or_respond",
        tool_condition,
        {
            "retrieve": "tool_execution",
            "respond": END
        }
    )
    graph.add_edge("tool_execution", "retrieve_or_respond")
    # return the compiled graph
    return graph.compile()