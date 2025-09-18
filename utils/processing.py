from langgraph.graph.state import CompiledStateGraph
from PIL import Image
import io
from pathlib import Path
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from utils.state import AgentState

def get_topics(folder_path: str):
    store_dir = Path(folder_path)

    if not store_dir.exists():
        raise FileNotFoundError(f"{store_dir} is not a existing directory")
    
    topics=[]
    for dir in store_dir.iterdir():
        topics.append(dir.name)

    return topics

def save_to_png(agent: CompiledStateGraph):
    img_data = agent.get_graph().draw_mermaid_png()
    img = Image.open(io.BytesIO(img_data))
    img.save("graph.png")

def stream_response(agent: CompiledStateGraph[AgentState], user_query: str):
    for event in agent.stream({"messages": [HumanMessage(user_query)], "question": user_query, "context": ""}):
        for value in event.values():
            value["messages"][-1].pretty_print()
            if "question" in value:
                question = value["question"]
                print(f"\nQuestion: {question}")
            if "context" in value:
                context = (value["context"][:100] + "...") if value["context"] else ""
                print(f"Context: {context}")