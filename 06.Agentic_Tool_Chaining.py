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


load_dotenv()  # Load environment variables from .env file
groq_api_key = os.getenv("groq_api")

# LLM decides dynamically

# Search
#   ↓
# Analyze Results
#   ↓
# Maybe Search Again
#   ↓
# Summarize
#   ↓
# Generate Report


llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=1000,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


# AGENTIC TOOL CHANING: Agent descides sequence
# // Websearch
search = DuckDuckGoSearchRun()

@tool
def search_tool(query: str) -> str:
    """
    Search the web for current information about a topic.
    
    This tool performs a real-time web search using DuckDuckGo search engine
    and returns relevant search results for the given query.
    
    Args:
        query (str): The search query or topic to find information about.
                     Examples: "AI", "Python programming", "weather in London"
    
    Returns:
        str: Search results containing relevant information about the query.
             Results are formatted as concatenated text from top matches.
    
    Raises:
        Exception: If the search fails or query is empty.
    
    Examples:
        >>> search_tool("artificial intelligence")
        "Artificial intelligence (AI) is..."
        
        >>> search_tool("Python tutorials")
        "Learn Python programming from..."
    
    Note:
        - Requires internet connection
        - Results are sourced from DuckDuckGo
        - Returns raw search results without processing
    """
    return search.run(query)


@tool
def summarize_tool(text: str) -> str:
    """
    Summarize text into a concise one-line summary.
    
    This tool uses the LLM to condense any given text into a single line
    summary that captures the main idea or key information.
    
    Args:
        text (str): The text content to be summarized.
                    Can be any length - from paragraphs to long documents.
    
    Returns:
        str: A concise one-line summary of the input text.
    
    Raises:
        Exception: If the LLM fails to generate a summary.
    
    Examples:
        >>> summarize_tool("Artificial intelligence is the capability of computational systems...")
        "AI is technology that enables machines to perform human-like tasks."
        
        >>> summarize_tool("Python is a high-level programming language...")
        "Python is a versatile, easy-to-learn programming language."
    
    Note:
        - Summary is always constrained to one line
        - Uses LLM for intelligent condensing
        - Best used after search_tool to compress search results
    """
    prompt = f"Summarize this text into ONE LINE: {text}"

    response = llm.invoke(prompt)

    return response.content

@tool
def translate_tool(text: str) -> str:
    """
    Translate text into Telugu language.
    
    This tool uses the LLM to translate any given text from the current language
    (typically English) into Telugu, an Indian language spoken in South India.
    
    Args:
        text (str): The text to be translated to Telugu.
                    Can be a word, phrase, sentence, or paragraph.
    
    Returns:
        str: The input text translated into Telugu language.
    
    Raises:
        Exception: If the LLM fails to generate a translation.
    
    Examples:
        >>> translate_tool("Hello")
        "హలో"
        
        >>> translate_tool("Good morning")
        "శుభోదయం"
    
    Note:
        - Translation quality depends on LLM capabilities
        - Best results for simple phrases and sentences
        - Can be used as final step after search + summarize
        - Telugu script uses Unicode characters
    """
    prompt=f"Translate this into Telugu language: {text}"

    response = llm.invoke(prompt)

    return response.content 

tools = [search_tool, summarize_tool, translate_tool]

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with THREE tools available:
1. search_tool - searches the web for information about any topic
2. summarize_tool - condenses text into one-line summaries
3. translate_tool - translates text to Telugu

INSTRUCTIONS:
- If user asks for information: use search_tool with their query
- If user asks to summarize text: use summarize_tool with the text
- If user asks to translate to Telugu: use translate_tool with the text
- If user wants multiple operations (search + summarize + translate), call tools in sequence
- Always return the final result to the user in a clear way

Examples:
- "find info about X" -> call search_tool("X")
- "summarize this text" -> call summarize_tool("the text")
- "convert X to Telugu" -> call translate_tool("X")
- "find about X, summarize and convert to Telugu" -> search_tool -> summarize_tool -> translate_tool"""),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}")
])

agent = create_tool_calling_agent(llm_with_tools, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)




while True:
    input_que = input("Enter text (or 'exit' to quit):")
    if input_que.lower() == 'exit':
        break
    
    result = agent_executor.invoke(
        {"input": input_que}
    )

    output = result.get("output", "No response generated")
    print("Response:", output)