from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools.retriever import create_retriever_tool
from pathlib import Path
import json

from utils import get_topics, save_to_png, stream_response



if __name__ == "__main__":

    if not Path("./config/app_config.json").exists():
        raise FileNotFoundError("app_config.json not Found")
    
    if not Path("./config/prompts.json").exists():
        raise FileNotFoundError("prompts.json not Found")

    with open("./config/app_config.json") as f:
        app_config = json.load(f)
    
    with open("./config/prompts.json") as f:
        prompts = json.load(f)

    topics = get_topics(app_config["db_dir_path"])

    # graph = StateGraph(AgentState)


    # agent = graph.compile()

    # # save_to_png(agent)
    # user_query = input("Enter: ")
    
    # stream_response(agent, user_query)