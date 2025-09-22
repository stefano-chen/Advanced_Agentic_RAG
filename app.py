from pathlib import Path
import json
from utils.processing import save_to_png, stream_response
from utils.agent import build_agent
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
import time


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
    start = time.time()
    agent = build_agent(app_config, prompts)
    end = time.time()
    print(f"Agent compiled in {(end - start):.2f} seconds")

    verbosity = app_config.get("verbosity", 0)
    save_to_png_flag = app_config.get("save_to_png", False)
    image_name = app_config.get("image_name", "graph.png")
    if save_to_png_flag:
        save_to_png(agent, image_name)

    user_query = input("Enter: ")
    while user_query.lower() not in ["exit", "quit"]:
        history = "\n".join(msg.content for msg in chat_history.messages)
        chat_history.add_user_message(user_query)
        start = time.time()
        answer = stream_response(agent, user_query, history, verbosity)
        end = time.time()
        chat_history.add_ai_message(answer)
        print(f"\n{'-'*30} Answer({(end-start):.2f}s) {'-'*30}\n{answer}")
        user_query = input("Enter: ")