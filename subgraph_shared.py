from langgraph.graph import StateGraph, START,END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm_parent = ChatOpenAI("gpt-4o")
llm_subgraph = ChatOpenAI("gpt-4o-mini")

class SharedState(TypedDict):
    topic: str
    blog_text: str
    translated_text: str

def translate_blog(state: SharedState):
    prompt = f"""You are helpful assistant to convert provided statement to Hindi text. 
    Translation needs to be neat and clean. 
    Do not add any extra line. 
    Statement: {state['blog_text']}"""
    
    response = llm_subgraph.invoke(prompt).content
    
    return {"translated_text" : response}

sub_graph = StateGraph(SharedState)

sub_graph.add_node("translate_blog",translate_blog)

sub_graph.add_edge(START,"translate_blog")
sub_graph.add_edge("translate_blog",END)

subgraph = sub_graph.compile()

def generate_blog(state: SharedState):
    prompt = f"""Generate a blog on user provided topic. Blog must have 100 words long only.
                Topic: {state.get("topic")}
                """
                
    blog = llm_parent.invoke(prompt).content
    
    return {"blog_text" : blog}

graph = StateGraph(SharedState)

graph.add_node("generate_blog",generate_blog)
graph.add_node("translate_blog",subgraph)

graph.add_edge(START,"generate_blog")
graph.add_edge("generate_blog","translate_blog")
graph.add_edge("translate_blog",END)

workflow = graph.compile()

initial_state = {"topic": "Write a blog on tiger."}
result = workflow.invoke(initial_state)

print(result)