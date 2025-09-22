from langgraph.graph.state import CompiledStateGraph
from PIL import Image
import io
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from utils.state import AgentState
from typing import List, Union
from langchain_core.runnables.graph import MermaidDrawMethod

def get_topics(folder_path: str):
    store_dir = Path(folder_path)

    if not store_dir.exists():
        raise FileNotFoundError(f"{store_dir} is not a existing directory")
    
    topics=[]
    for dir in store_dir.iterdir():
        topics.append(dir.name)

    return topics

def save_to_png(agent: CompiledStateGraph, file_name: str):
    img_data = agent.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
    img = Image.open(io.BytesIO(img_data))
    img.save(file_name)

def stream_response(agent: CompiledStateGraph[AgentState], user_query: str, history: List[Union[AIMessage, HumanMessage]],  verbosity: int = 0):

    last_msg = None
    prefix = "\n" if verbosity > 0 else ""

    for event in agent.stream(AgentState.create(messages=[HumanMessage(user_query)], question=user_query, history=history)):
        for key, value in event.items():
            print(f"{prefix}STEP: {key}", flush=True)
            last_msg =  value["messages"][-1]
            if verbosity > 0:
                last_msg.pretty_print()
                if verbosity > 1:
                    if last_msg.type != "tool":
                        print(f"\n{'-'*36} STATE {'-'*37}")
                        print(f"Question: {value["question"]}")
                        print(f"Original Question: {value["original_question"]}")
                        context = (value["context"][:100] + "...") if value["context"] else ""
                        print(f"Context: {context}")
                        print(f"Reranking score: {value["reranking_score"]}")
                        chunks = (str(value["chunks"])[:100] + "...]") if value["chunks"] else value["chunks"]
                        print(f"Chunks: {chunks}")
                        print(f"{'-'*80}")
    
    answer = last_msg.content

    print(f"\n{'-'*36} Answer {'-'*36}\n{answer}")

    return answer