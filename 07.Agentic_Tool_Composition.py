# The easiest way to see the difference is:

# Tool Chaining	                 Tool Composition
# Connect tools in a sequence	 Combine tools into a larger tool/capability
# Focus is on execution flow	 Focus is on abstraction/reuse
# A → B → C	                     (A + B + C) = D
# User/developer sees each step	User sees one higher-level 

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableWithMessageHistory


load_dotenv()  # Load environment variables from .env file
groq_api_key = os.getenv("groq_api")


llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=1000,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


@tool
def google_search(query: str) -> str:
    """
    version: 1.0.0
    author: Development Team
    created: 2026-06-13
    last_updated: 2026-06-13
    
    description: |
        Search the web for information using DuckDuckGo search engine.
        This tool performs a real-time web search and returns relevant results.
    
    input:
        - name: query
          type: string
          required: Yes
          description: The search query string
          examples: 
            - "machine learning"
            - "Python programming"
            - "weather today"
          constraints:
            min_length: 1
            max_length: 500
    
    output:
        - name: result
          type: string
          description: Search results formatted as concatenated text
          examples:
            - "AI trends include... Model improvements enable..."
            - "Python methods: append, extend, insert, remove..."
          constraints:
            min_length: 0
            max_length: 5000
    
    error:
        - name: ValueError
          when: "Query is empty or exceeds 500 characters"
          example: "ValueError: Query must be 1-500 characters"
        
        - name: ConnectionError
          when: "Internet connection is unavailable"
          example: "ConnectionError: Search failed - No internet"
        
        - name: TimeoutError
          when: "Search request times out"
          example: "TimeoutError: Search took more than 30 seconds"
    
    notes: |
        - Requires internet connectivity
        - Response time: 2-10 seconds typical
        - Results cached for 1 hour
        - Results sourced from DuckDuckGo
        - Production ready: Yes
    
    usage_examples: |
        >>> google_search("artificial intelligence")
        >>> google_search("python lists tutorial")
    """
    search = DuckDuckGoSearchRun()
    try:
        if not query or len(query) > 500:
            raise ValueError("Query must be 1-500 characters")
        return search.run(query)
    except Exception as e:
        raise ConnectionError(f"Search failed: {str(e)}")
    

@tool
def summarizer(info: str) -> str:
    """
    version: 0.0.1
    author: someone
    created: 2026-06-13
    last_updated: 2026-06-13

    description: This tool summarizers the info that it gets.... Alwasy summarize into
    very short and crips in simple english format

    input: 
       - name: info
         type: string
         required: true
         description: This is the input that we summarize into short

    output:
       - name: result
         type: string
         required: true
         description: This is the output we sent back after summarizing
    error:
        - name: ValueError
          when: "Query is empty or exceeds 500 characters"
          example: "ValueError: Query must be 1-500 characters"
        
        - name: ConnectionError
          when: "Internet connection is unavailable"
          example: "ConnectionError: Search failed - No internet"
        
        - name: TimeoutError
          when: "Search request times out"
          example: "TimeoutError: Search took more than 30 seconds"

        Notes:
         1. Dont go beyond  3 lines
         2. Alwasy sarcastic of explaining things
         3. Dont be dumb
    
        """
    
    response = llm.invoke(f"Alwasy summarize into simple english:  {info}")

    return response.content

@tool
def translate_tool(info: str) -> str:
    """
    version: 0.0.1
    author: someo
    description: This tool would be usefult to translate into Telugu Language...
    """
    response = llm.invoke(info)

    return response.content


# Composed tool: hides the individual steps and exposes a single capability
@tool
def search_summarize_translate(query: str) -> str:
    """
    Composed tool that performs: search -> summarize -> translate.

    Input:
      - query (str): Search query string

    Output:
      - Telugu translation of a short summary of search results
    """
    # Step 1: search
    search_results = google_search.run(query)

    # Step 2: summarize
    summary = summarizer.run(search_results)

    # Step 3: translate
    translated = translate_tool.run(summary)

    return translated


tools  = [search_summarize_translate]

llm_with_tools = llm.bind_tools(tools)

store = {}

def get_session_by_id(id):
    if id in store:
        return store[id]
    else:
        store[id] = ChatMessageHistory()
        return store[id]


prompt_template =  ChatPromptTemplate.from_messages([
  ('system', """You are an AI Assistant. If the user asks for information plus summarization and translation, prefer the composed tool `search_summarize_translate` which performs search→summarize→translate internally. For other requests, respond directly or use the composed tool as appropriate. Keep responses short, clear, and politely sarcastic when asked."""),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}")
])


tool_calling_agent = create_tool_calling_agent(
    llm=llm_with_tools,
    prompt= prompt_template,
    tools=tools
)

agent_executor = AgentExecutor(
    agent = tool_calling_agent,
    tools = tools,
    verbose = True
)

agent_executor_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_by_id,
    input_messages_key="input",
    history_messages_key="history"
)


while True:
    inputq_  = input("Question (or 'exit' to quit): ")
    if inputq_.lower() == 'exit':
        break

    response = agent_executor_with_memory.invoke(
        {"input": inputq_},
        config={"configurable": {"session_id": "user_session"}}
    )
    print("\nResponse:", response.get("output", "No response"))
    print("-" * 80)