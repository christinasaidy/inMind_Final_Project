import os
from enum import Enum
from typing import List, Tuple
from pydantic import BaseModel, Field

##import my vectorstore
from articles_rag import article_vectorstore

from mistralai import Mistral
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool

## reference : https://github.com/mistralai/cookbook/blob/main/mistral/evaluation/RAG_evaluation.ipynb

RAG_QUESTIONS = [
    "What were the main drivers of Lebanon’s cost of living in 2024, and which expense categories were most affected?",
    "According to the World Bank’s Spring 2025 Lebanon Economic Monitor, how is dollarization shaping the economy and what structural constraints persist?",
    "What are the key elements of the World Bank’s 2025 roadmap to revitalize Lebanon’s economy, and which reforms are prioritized for sustainability?",
    "How did the liquidity crisis affect the banking sector and the Lebanese pound since 2019, and what mechanisms caused these effects?",
    "What role do external trade and investment (e.g., Italy/Lebanon ties) play in Lebanon’s recovery prospects?",
]

answer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyByctz3rxpT0W_s9bEV2lJUyXDDvXvP6Ys")

judge_client = Mistral(api_key="QsEVKiR9jCLqvfLnItqHLKMQGes0Te8Q")
JUDGE_MODEL = "mistral-large-latest"


class Score(str, Enum):
    no_relevance = "0"
    low_relevance = "1"
    medium_relevance = "2"
    high_relevance = "3"

SCORE_DESCRIPTION = (
    "Score as a string between '0' and '3'. "
    "0: No relevance/Not grounded/Irrelevant - The context/answer is completely unrelated or not based on the context. "
    "1: Low relevance/Low groundedness/Somewhat relevant - The context/answer has minimal relevance or grounding. "
    "2: Medium relevance/Medium groundedness/Mostly relevant - The context/answer is somewhat relevant or grounded. "
    "3: High relevance/High groundedness/Fully relevant - The context/answer is highly relevant or grounded."
)

class ContextRelevance(BaseModel):
    explanation: str = Field(
        ...,
        description="Step-by-step reasoning explaining how the retrieved context aligns with the user's query."
    )
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class AnswerRelevance(BaseModel):
    explanation: str = Field(
        ...,
        description="Step-by-step reasoning explaining how well the generated answer addresses the user's original query."
    )
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class Groundedness(BaseModel):
    explanation: str = Field(
        ...,
        description="Step-by-step reasoning explaining how faithful the generated answer is to the retrieved context."
    )
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class RAGEvaluation(BaseModel):
    context_relevance: ContextRelevance
    answer_relevance: AnswerRelevance
    groundedness: Groundedness


def evaluate_rag(query: str, retrieved_context: str, generated_answer: str) -> RAGEvaluation:
    """
    Calls Mistral with structured output, returning your Pydantic RAGEvaluation.
    """
    chat_response = judge_client.chat.parse(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a judge for evaluating a Retrieval-Augmented Generation (RAG) system. "
                    "Evaluate the context relevance, answer relevance, and groundedness based on the following criteria: "
                    "Provide a reasoning and a score as a string between '0' and '3' for each criterion. "
                    "Context Relevance: How relevant is the retrieved context to the query? "
                    "Answer Relevance: How relevant is the generated answer to the query? "
                    "Groundedness: How faithful is the generated answer to the retrieved context?"
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\nRetrieved Context: {retrieved_context}\nGenerated Answer: {generated_answer}"
            },
        ],
        response_format=RAGEvaluation,
        temperature=0
    )
    return chat_response.choices[0].message.parsed


def answer_with_gemini(query: str) -> str:
 
    msg = answer_llm.invoke([
        SystemMessage(content="Answer wisely."),
        HumanMessage(content=query),
    ])
    return msg.content if hasattr(msg, "content") else str(msg)


def answer_with_rag(query: str, k: int = 5) -> Tuple[str, str]:

    retriever = article_vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(retriever,"retrieve_budget_info", query)
    retrieved_context = retriever_tool.run(query)

    prompt = (
        "Use ONLY the following context to answer. If the information is missing, say you don't have enough data.\n\n"
        f"### Context\n{retrieved_context}\n\n"
        f"### Question\n{query}"
    )
    msg = answer_llm.invoke([
        SystemMessage(content="Cite from the provided context only; "),
        HumanMessage(content=prompt),
    ])
    answer = msg.content if hasattr(msg, "content") else str(msg)
    return answer, retrieved_context


##prety print from chatgpt

def print_eval(title: str, eval_obj: RAGEvaluation):
    print(f"\n[{title}]")
    print("Context Relevance:", eval_obj.context_relevance.score.value, "/3")
    print("  ↳", eval_obj.context_relevance.explanation)
    print("Answer Relevance:", eval_obj.answer_relevance.score.value, "/3")
    print("  ↳", eval_obj.answer_relevance.explanation)
    print("Groundedness:", eval_obj.groundedness.score.value, "/3")
    print("  ↳", eval_obj.groundedness.explanation)


def main():
    for i, q in enumerate(RAG_QUESTIONS, start=1):
        print("\n" + "=" * 100)
        print(f"Q{i}: {q}")

        base_answer = answer_with_gemini(q)
        base_eval = evaluate_rag(q, retrieved_context="", generated_answer=base_answer)

        print("\n-- Baseline (Gemini, no retrieval) --")
        print(base_answer)
        print_eval("Judge (Baseline)", base_eval)

        rag_answer, rag_context = answer_with_rag(q, k=6)
        rag_eval = evaluate_rag(q, retrieved_context=rag_context, generated_answer=rag_answer)

        print("\n-- RAG (Gemini + vector store) --")
        print(rag_answer)
        print("\n-- Retrieved Context (truncated) --")
        print(rag_context[:1500])  
        print_eval("Judge (RAG)", rag_eval)


if __name__ == "__main__":
    main()