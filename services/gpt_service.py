# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence
# import os
# from config import OPENAI_API_KEY

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# llm = ChatOpenAI(model="davinci-002", temperature=0)

# prompt = PromptTemplate(
#     input_variables=["query"],
#     template="""
# You are a smart multilingual product search assistant.
# Your task is to:
# 1. Detect the language of the query.
# 2. Translate it to English if needed.
# 3. Fix typos and grammar issues.
# 4. Rewrite it as a clean, optimized search query.

# Raw Query: {query}

# Optimized English Search Query:
# """
# )

# chain = prompt | llm  # RunnableSequence

# def optimize_query_with_gpt(raw_query):
#     return chain.invoke({"query": raw_query})

import cohere
co = cohere.Client("7vh0EuMhDhYc454iO8bQRQyUy51XVmsNMkPRoqQt")

def optimize_query_with_cohere(raw_query):
    prompt = f"""You are a smart multilingual product search assistant. 
Given this query: "{raw_query}", detect its language, translate to English, fix typos, and rewrite it as a clean product search query."""
    
    response = co.generate(prompt=prompt, model="command-r", max_tokens=50)
    return response.generations[0].text.strip()
