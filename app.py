from pathlib import Path
import json
from utils.processing import save_to_png, stream_response
from utils.agent import build_agent
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory


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

    chat_history = InMemoryChatMessageHistory()

    agent = build_agent(app_config, prompts)

    verbosity = app_config.get("verbosity", 0)
    save_to_png_flag = app_config.get("save_to_png", False)
    image_name = app_config.get("image_name", "graph.png")
    if save_to_png_flag:
        save_to_png(agent, image_name)

    user_query = input("Enter: ")
    while user_query.lower() not in ["exit", "quit"]:
        history = chat_history.messages
        chat_history.add_user_message(user_query)
        answer = stream_response(agent, user_query, history, verbosity)
        chat_history.add_ai_message(answer)
        user_query = input("Enter: ")