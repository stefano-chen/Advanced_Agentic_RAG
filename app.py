from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from pathlib import Path
import json
from utils.processing import get_topics, save_to_png, stream_response
from utils.llm import LLMModel
from nodes.query_validation import QueryValidation, is_related
from dotenv import load_dotenv
from nodes.tool_calling import ToolCalling, tool_condition
from tools.retrieval import get_tools

load_dotenv("./.env")

if __name__ == "__main__":

    if not Path("./config/app_config.json").exists():
        raise FileNotFoundError("app_config.json not Found")
    
    if not Path("./config/prompts.json").exists():
        raise FileNotFoundError("prompts.json not Found")

    with open("./config/app_config.json") as f:
        app_config = json.load(f)
    
    with open("./config/prompts.json") as f:
        prompts = json.load(f)

    llm = LLMModel(app_config["llm_provider"], app_config["llm_model"], app_config["llm_host"]).get()

    topics = get_topics(app_config["db_dir_path"])

    tools = get_tools(app_config["vector_db"], app_config["db_dir_path"], app_config["k"], app_config["embedding_provider"], app_config["embedding_model"], app_config["embedding_host"])

    graph = StateGraph(MessagesState)

    # Insert Graph Node and Edges
    graph.add_node("validate_input", QueryValidation(llm, prompts["input_check"], topics).validate)
    graph.add_node("retrieve_or_respond", ToolCalling(llm, prompts["retrieve_respond"], tools).choose)
    graph.add_node("tool_execution", ToolNode(tools))

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
    agent = graph.compile()

    save_to_png(agent)
    user_query = input("Enter: ")
    
    stream_response(agent, user_query)