"""
Simple usage example - closest to the original embeder.py usage
Now uses .env file for configuration
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_rag.indexing import IndexBuilder
from code_rag.retrieval import AnswerGenerator
from code_rag.config import get_indices_dir

if __name__ == "__main__":
    builder = IndexBuilder(
        root_path=r"/Users/mjraei/Desktop/Projects/OplaSmart/engReports"
    )
    
    builder.build_index()
    
    index_path = os.path.join(get_indices_dir(), "reporting_module_index")
    builder.save_index(index_path)
    
    answer_gen = AnswerGenerator(
        index_path=index_path
    )
    
    result = answer_gen.ask_question("How the pumps output is calculated in the reports?")
    
    print("ANSWER:")
    print(result['answer'])
    print(f"\nSource files: {', '.join(result['source_files'])}")
    print(f"Functions mentioned: {', '.join(result['functions_mentioned'])}")
    
    # Example of follow-up question
    # conversation = [
    #     {"role": "user", "content": "How the pumps output is calculated in the reports?"},
    #     {"role": "assistant", "content": result['answer']}
    # ]
    # 
    # follow_up = answer_gen.chat(conversation, "What parameters does it accept?")
    # print("\nFOLLOW-UP ANSWER:")
    # print(follow_up['answer']) 