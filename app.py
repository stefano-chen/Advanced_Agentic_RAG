from pathlib import Path
import json
from utils.processing import get_topics, save_to_png, stream_response
from utils.llm import LLMModel
from tools.retrieval import get_tools
from utils.agent import build_agent
from dotenv import load_dotenv


if __name__ == "__main__":

    load_dotenv("./.env")

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

    agent = build_agent(llm, prompts, tools, topics)

    save_to_png(agent)
    user_query = input("Enter: ")
    
    stream_response(agent, user_query)