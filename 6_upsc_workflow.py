from langgraph.graph import StateGraph, START,END
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import operator
import json
import re
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI()

essay="""The Pervasive Presence of AI in Today's World

Artificial Intelligence (AI) has revolutionized the way we live, work, and interact with one another. In recent years, AI has become an integral part of our daily lives, transforming the world in ways both subtle and profound. From the simplest tasks to the most complex operations, AI has made its presence felt across various sectors, redefining the boundaries of what is possible.

The Rise of AI in Everyday Life

One of the most noticeable aspects of AI's impact is the proliferation of virtual assistants like Siri, Google Assistant, and Alexa. These AI-powered tools have made it easier for people to access information, set reminders, and control their smart home devices with just their voices. Moreover, AI-driven chatbots have become a common feature on websites and mobile apps, providing instant customer support and helping businesses automate their customer service.

Transforming Industries

AI has also had a significant impact on various industries, including healthcare, finance, and transportation. In healthcare, AI algorithms are being used to analyze medical images, diagnose diseases, and develop personalized treatment plans. For instance, AI-powered computer vision is being used to detect breast cancer from mammography images, while natural language processing (NLP) is being applied to analyze patient data and identify potential health risks.

In the financial sector, AI is being used to detect fraudulent transactions, predict market trends, and provide personalized investment advice. AI-powered chatbots are also being used to help customers with their queries and provide support.

The transportation sector has also seen a significant transformation with the advent of AI. Self-driving cars, powered by AI, are being tested and implemented on roads around the world. These vehicles use a combination of sensors, GPS, and AI algorithms to navigate through traffic, avoid accidents, and optimize routes.

The Dark Side of AI

While AI has brought many benefits, it also raises concerns about job displacement, bias, and security. As AI automates routine tasks, there is a risk that many jobs will become redundant, leading to significant social and economic disruption. Moreover, AI systems can perpetuate biases present in the data used to train them, leading to unfair outcomes in areas like hiring, law enforcement, and lending.

The Future of AI

As AI continues to evolve, we can expect to see even more innovative applications across various sectors. For instance, AI is likely to play a critical role in addressing some of the world's most pressing challenges, such as climate change, disease diagnosis, and sustainable development.

However, to ensure that AI benefits society as a whole, it is essential to develop and deploy AI systems that are transparent, explainable, and fair. This requires a multidisciplinary approach, involving experts from fields like computer science, ethics, law, and social sciences.

Conclusion

In conclusion, AI has become an integral part of our world, transforming the way we live, work, and interact with one another. While AI has brought many benefits, it also raises concerns about job displacement, bias, and security. As AI continues to evolve, it is essential to prioritize responsible AI development and deployment, ensuring that its benefits are equitably distributed and its risks are mitigated. Ultimately, the future of AI holds much promise, and it is up to us to shape it in a way that benefits humanity as a whole.

Recommendations

Invest in AI education and training: To prepare workers for an AI-driven economy, governments and businesses must invest in education and training programs that focus on developing AI-related skills.
Develop transparent and explainable AI systems: Developers must prioritize transparency and explainability in AI system design, ensuring that decisions made by AI systems are understandable and fair.
Address AI bias and security concerns: Developers and regulators must work together to address AI bias and security concerns, ensuring that AI systems are fair, secure, and respect human rights.
Foster international cooperation on AI governance: Governments and international organizations must work together to develop common standards and guidelines for AI development and deployment, ensuring that AI benefits humanity as a whole.
By prioritizing responsible AI development and deployment, we can ensure that AI continues to transform the world for the better, improving lives, and creating new opportunities for growth and innovation."""

prompt = f'Evaluate the language quality of the following eassy and provide a feedback and assign a score out of 10. \n {essay}.\n your response must be in valid JSON format as key "feeback" and "score".\n Stricly do not add any pre or post statement to the response.' 


class UPSCState(TypedDict):
    essay:str
    language_feedback: str
    analysis_feedback:str
    thought_feedback:str
    final_feedback:str

    individual_scores: Annotated[list[int], operator.add]
    final_score: float

def parse_llm_json_response(response: str) -> dict:
    cleaned_response = response.strip()
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
        if cleaned_response.startswith("json"):
            cleaned_response = cleaned_response[4:]
        cleaned_response = cleaned_response.strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()
    json_start = cleaned_response.find("{")
    json_end = cleaned_response.rfind("}")
    if json_start != -1 and json_end != -1:
        cleaned_response = cleaned_response[json_start:json_end + 1]
    cleaned_response = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", cleaned_response)
    return json.loads(cleaned_response, strict=False)


def evaluate_language(state: UPSCState):
    prompt = f'Evaluate the language quality of the following eassy and provide a feedback and assign a score out of 10. \n {state.get("essay")}.\n your response must be in valid JSON format as key "feeback" and "score".\n Stricly do not add any pre or post statement to the response.'

    response = model.invoke(prompt).content
    language_feedback = parse_llm_json_response(response)
    print(f"feedback: {language_feedback["feedback"]}")
    print(f"individual_scores: {language_feedback["score"]}")
    print(f"individual_scores: {[language_feedback["score"]]}")
    return {"language_feedback": language_feedback["feedback"], "individual_scores": [language_feedback["score"]]}

def evaluate_thought(state: UPSCState):
    prompt = f'Evaluate the depth of analysis of the following eassy and provide a feedback and assign a score out of 10. \n {state.get("essay")}.\n your response must be in valid JSON format as key "feeback" and "score".\n Stricly do not add any pre or post statement to the response.'

    response = model.invoke(prompt).content
    print(f"before thought response: {response}")
    thought_feedback = parse_llm_json_response(response)
    print(f"thought response: {thought_feedback}")
    return {"thought_feedback": thought_feedback["feedback"], "individual_scores": [thought_feedback["score"]]}


def evaluate_analysis(state: UPSCState):
    prompt = f'Evaluate the clarity of thought of the following eassy and provide a feedback and assign a score out of 10. \n {state.get("essay")}.\n your response must be in valid JSON format as key "feeback" and "score".\n Stricly do not add any pre or post statement to the response.'

    response = model.invoke(prompt).content
    analysis_feedback = parse_llm_json_response(response)
    return {"analysis_feedback": analysis_feedback["feedback"], "individual_scores": [analysis_feedback["score"]]}


def final_evaluation(state: UPSCState):
    prompt = f'Based on the following feedbacks create a summarized feedback.\n Language feedback: {state.get("language_feedback")}'
    #\n depth of analysis feedback: {state.get("analysis_feedback","")}\n clarity of thoughts feedback: {state.get("thought_feedback","")}
    response = model.invoke(prompt).content

    print(f"state: {state}")
    overall_score = sum(state.get("individual_scores"))/len(state.get("individual_scores"))

    return {"final_feedback": response, "final_score": overall_score}


graph = StateGraph(UPSCState)

graph.add_node("evaluate_language",evaluate_language)
#graph.add_node("evaluate_thought",evaluate_thought)
#graph.add_node("evaluate_analysis",evaluate_analysis)
graph.add_node("final_evaluation",final_evaluation)

graph.add_edge(START,"evaluate_language")
#graph.add_edge(START,"evaluate_thought")
#graph.add_edge(START,"evaluate_analysis")

graph.add_edge("evaluate_language","final_evaluation")
#graph.add_edge("evaluate_thought","final_evaluation")
#graph.add_edge("evaluate_analysis","final_evaluation")

graph.add_edge("final_evaluation",END)

workflow=graph.compile()


initial_state = {
    "essay": essay
}
final_state = workflow.invoke(initial_state)
print(final_state)