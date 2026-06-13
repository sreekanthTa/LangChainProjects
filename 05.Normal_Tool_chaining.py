from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
groq_api_key = os.getenv("groq_api")


# Developer decides

# Search
#   ↓
# Summarize
#   ↓
# Translate

# Always the same.


llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=1000,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

search = DuckDuckGoSearchRun()

# -----------------------
# Tools (simulated)
# -----------------------
def search_tool(query: str) -> str:
    return search.run(query)

def summarize_tool(text: str) -> str:
    return llm.invoke(f"Summraize into small: {text}")

def translate_tool(text: str) -> str:
    return llm.invoke(f"Convert to Hindi :{text}")


# -----------------------
# TOOL CHAIN (MANUAL ORCHESTRATION)
# -----------------------
def tool_chain(query: str) -> str:
    step1 = search_tool(query)        # YOU call first tool
    step2 = summarize_tool(step1)     # YOU pass output manually
    step3 = translate_tool(step2)     # YOU continue flow
    return step3


# -----------------------
# RUN
# -----------------------
result = tool_chain("latest AI news")
print(result)