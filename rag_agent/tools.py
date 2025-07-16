"""Tools for the RAG agent."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_rag.retrieval.answer_generator import AnswerGenerator

def query_codebase(question: str) -> str:
    """
    Answers questions about the codebase. 
    Use this tool to find information about the code, how it works, what different parts of it do, etc.
    For example: "How are pump outputs calculated?"

    Args:
        question (str): The question to ask about the codebase.

    Returns:
        str: The answer to the question.
    """
    print(f"Querying codebase with question: {question}")
    index_path = os.path.join("data", "reporting_module_index")
    
    if not os.path.exists(index_path + ".faiss"):
        return f"Error: Index not found at {index_path}. Please build the index first."

    try:
        answer_gen = AnswerGenerator(index_path=index_path)
        result = answer_gen.ask_question(question)
        return result['answer']
    except Exception as e:
        return f"An error occurred while querying the codebase: {e}" 