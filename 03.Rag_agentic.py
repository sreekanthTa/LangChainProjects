from langchain_core.prompts import ChatPromptTemplate
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# message  place holder
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()  # Load environment variables from .env file
groq_api_key = os.getenv("groq_api")


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_docs")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

store = {}

def get_session_history_by_id(id):
    if id in store:
        return store[id]
    else:
        store[id] = ChatMessageHistory()
        return store[id]

@tool
def search_pdf(query: str) -> str:
    """
    version: 1.0
    author: somename
    name: search_pdf
    content: About Metformin Information
    description: Search for relevant information in the PDF based on the query and return the context.
    inputs:
      - name: query
        type: string
        description: The question or query for which you want to search the PDF.
    
    outputs:
        - name: context
          type: string
          description: The relevant context retrieved from the PDF based on the query.

    """
    print("Tool callling with query:", query)

    query_embedding = embed_model.encode(query, normalize_embeddings=True)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    context = ""

    for i, doc in enumerate(results['documents'][0]):
        context += doc + "\n"

    return context


tools =  [search_pdf]

llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=300,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

llm=llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer in Sarcastic Comedy"),
    MessagesPlaceholder(variable_name="history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}")
])


# chain = prompt | llm


agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True
)

agent_executor_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history_by_id,
    input_messages_key="input",
    history_messages_key="history"
)

while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    response = agent_executor_with_memory.invoke(
        {"input": question},
        config={"configurable": {"session_id": "user1"}}
    )
    print("LLM Response:", response.get("output"))