from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.graph import START,StateGraph,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt,Command
from dotenv import load_dotenv
from typing import TypedDict ,Annotated

load_dotenv()
llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState) -> ChatState:
    decision = interrupt({
        "type" : "approval",
        "reason": "model is about the answer user's asked question.",
        "question": state["messages"][-1].content,
        "instruction": "Aprove this question? yes/no"
    })
    
    if decision["user_input"] == "no":
        return  {"messages" : [AIMessage(content="Not Approved.")]}
    else:
        response = llm.invoke([HumanMessage(state["messages"])])
        return {"messages": [response]}

graph = StateGraph(ChatState)

graph.add_node("chat_node",chat_node)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

checkpointer = MemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

initial_state = {"messages" : [ ("user", "What is the capital of India?")]}

config = {"configurable" : {"thread_id": "thread-1"}}

result = workflow.invoke(initial_state,config=config)

print(result)