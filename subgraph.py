from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

#state
class SubgraphState(TypedDict):
    input_text: str
    translated_text: str
    
#Node
def translate(state: SubgraphState) -> SubgraphState:
    prompt = f"""You are helpful assistant to convert provided statement to Hindi text. 
    Translation needs to be neat and clean. 
    Do not add any extra line. 
    Statement: {state['input_text']}"""
    
    response = llm.invoke(prompt).content
    
    return {"translated_text" : response}

graph = StateGraph(SubgraphState)

graph.add_node("translate",translate)

graph.add_edge(START,'translate')
graph.add_edge('translate',END)

subgraph = graph.compile()

#initial_state = {"input_text" :"Learning something new every day keeps the mind active and curious. Whether it is reading a book, observing nature, or solving a small problem, each experience adds value. Consistency matters more than perfection. Even small steps can lead to meaningful growth over time. Staying patient and open to change helps in adapting to new challenges and opportunities."}
#subresult = subgraph.invoke(initial_state)

##### main workflow
main_flow = ChatOpenAI()

class MainState(TypedDict):
    topic: str
    blog_text: str
    translated_blog_text: str
    
def blog_generate(state: MainState):
    prompt = f"""Generate a blog on user provided topic. Blog must have 100 words long only.
                Topic: {state.get("topic")}
                """
                
    blog = main_flow.invoke(prompt)
    
    return {"blog_text" : blog}

def blog_translate(state: MainState):
    
    subgraph_result = subgraph.invoke({"input_text":state.get("blog_text")})
    
    translated_text = subgraph_result.get("translated_text")
    
    return {"translated_blog_text" : translated_text}

main_graph = StateGraph(MainState)

main_graph.add_node('blog_generate',blog_generate)
main_graph.add_node('blog_translate',blog_translate)

main_graph.add_edge(START,'blog_generate')
main_graph.add_edge('blog_generate','blog_translate')
main_graph.add_edge('blog_translate',END)

MainGraph = main_graph.compile()


initial_state = {"topic" : "Write a blog on tiger."}
result = MainGraph.invoke(initial_state)
print(f"result: {result}")