from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from nodes.query_validation import QueryValidation, is_related
from nodes.tool_calling import ToolCalling, tool_condition
from utils.state import AgentState
from nodes.query_transformation import QueryTransform
from utils.processing import get_topics
from tools.retrieval import get_tools
from utils.llm import LLMModel
from nodes.update_context import update_context
from nodes.answer import GenerateAnswer
from nodes.output_validation import AnswerValidation
from nodes.reranking import Reranking
from nodes.selection import ChunckSelection
from nodes.tool_calling import Choice
from nodes.extract_chunks import extract_chunks

def build_agent(app_config, prompts):

    check_output_validity_flag = app_config.get("check_output_validity", True)
    check_input_validity_flag = app_config.get("check_input_validity", True)
    advanced_rag_flag = app_config.get("advanced_rag", True)

    topics = get_topics(app_config["db_dir_path"])

    tools = get_tools(app_config["vector_db"], app_config["db_dir_path"], app_config["k"], app_config["embedding"])

    llm = LLMModel(app_config["llm"]).get()

    graph = StateGraph(AgentState)
    
    # Simple RAG Nodes
    graph.add_node("retrieve_or_respond", Choice(llm, prompts["retrieve_respond"]).choose)
    graph.add_node("tool_routing", ToolCalling(llm, prompts["tool_calling"], tools).call)
    graph.add_node("tool_execution", ToolNode(tools))
    graph.add_node("extract_chunks", extract_chunks)
    graph.add_node("update_context", update_context)
    graph.add_node("generate_answer", GenerateAnswer(llm, prompts["output"]).generate_answer)

    # Advanced RAG Nodes
    if advanced_rag_flag:
        if check_input_validity_flag:
            graph.add_node("validate_input", QueryValidation(llm, prompts["input_check"], topics).validate)
        graph.add_node("query_transform", QueryTransform(app_config["query_transform"], app_config["query_transform_options"], llm, prompts["query_transformation"]).transform)
        graph.add_node("reranking", Reranking(app_config["reranking_strategies"], app_config["reranking_weights"], app_config["reranking_strategies_options"], prompts).rerank)
        graph.add_node("selection", ChunckSelection(app_config["selection_strategies"], app_config["selection_options"]).select)
        if check_output_validity_flag:
            graph.add_node("validate_answer", AnswerValidation(llm, prompts["output_check"]).validate)

    # Always Present Edges
    graph.add_edge("retrieve_or_respond", "tool_routing")
    graph.add_conditional_edges(
        "tool_routing",
        tool_condition,
        {
            "retrieve": "tool_execution",
            "respond": "generate_answer"
        }
    )
    graph.add_edge("tool_execution", "extract_chunks")
    graph.add_edge("update_context", "retrieve_or_respond")

    if advanced_rag_flag:
        # Advanced RAG Edge
        if check_input_validity_flag:
            graph.add_edge(START, "validate_input")
            graph.add_conditional_edges(
                "validate_input",
                is_related,
                {
                    "yes": "query_transform",
                    "no": END
                }
            )
        else:
            graph.add_edge(START, "query_transform")
        graph.add_edge("query_transform", "retrieve_or_respond")
        graph.add_edge("extract_chunks", "reranking")
        graph.add_edge("reranking", "selection")
        graph.add_edge("selection", "update_context")
        if check_output_validity_flag:
            graph.add_edge("generate_answer", "validate_answer")
            graph.add_edge("validate_answer", END)
        else:
            graph.add_edge("generate_answer", END)
    else:
        # Simple RAG Edge
        graph.add_edge(START, "retrieve_or_respond")
        graph.add_edge("generate_answer", END)
        graph.add_edge("extract_chunks", "update_context")

    # return the compiled graph
    return graph.compile()