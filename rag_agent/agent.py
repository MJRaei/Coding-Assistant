"""Defines the RAG agent."""

from google.adk.agents import Agent
from .tools import query_codebase

root_agent = Agent(
   model="gemini-1.5-flash-latest",
   name="codebase_qa_agent",
   description="An agent that can answer questions about a codebase.",
   instruction="""
        You are an expert software engineer and your job is to answer questions about a codebase.
        When you are asked a question, use your `query_codebase` tool to find the answer.
        If the user is just greeting you, greet them back.
        Provide the answer in a clear and concise way.
        """,
   tools=[query_codebase],
) 