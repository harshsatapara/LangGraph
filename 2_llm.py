from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOCIGenAI
from typing import TypedDict
from langchain_openai import ChatOpenAI

class QAState(TypedDict):
    question: str
    answer: str

def generateAnswer(state: QAState) -> QAState:
    prompt = state.get("question","")    
    # initialize interface
    chat = ChatOpenAI()

    messages = [
    HumanMessage(content=prompt),
    ]


    response = chat.invoke(messages)

    if isinstance(response, AIMessage):
        print(f"response: {response.content}")
        response_json = response.content
        print(f"response_json: {response_json}")
    else:
        print(f"response: {response}")
        response_json = repr(response)
        print(f"response_json: {response_json}")
    
    state["answer"] = response_json
    return state
    
graph = StateGraph(QAState)
graph.add_node('generateAnswer',generateAnswer)

graph.add_edge(START,"generateAnswer")
graph.add_edge("generateAnswer",END)

workflow = graph.compile()

intial_state = {"question": "What it the capital of India?"}
final_state= workflow.invoke(intial_state)
print(f"final_state: {final_state}")