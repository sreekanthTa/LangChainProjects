from langchain_core.prompts import ChatPromptTemplate
import os
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# message  place holder
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory


load_dotenv()  # Load environment variables from .env file
groq_api_key = os.getenv("groq_api")


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_docs")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")




store = {}

def get_session_by_id(id):
    if id in store:
        return store[id]
    else:
        store[id] = ChatMessageHistory()
        return store[id]

@tool
def hybrid_search_pdf(query: str) -> str:
    """
    version:0.0.1
    owner: me
    input: 
        - name: query
          type: str
          description: Query that we need to find embedding
    output: 
        - name: context
          type: str
          description: Context of embeedding
    description: This searches pdf about the Metaformin in the similarity search 
                 and also keyword search and combines into hybrid search
    """
    # Semantic Retreval
    query_embedding=embed_model.encode([query], normalize_embeddings=True)

    response = collection.query(
        query_embeddings=query_embedding,
        n_results= 3
    )

    vector_docs = response["documents"][0]

    
    # Keyword Reterval
    all_docs =  collection.get()["documents"]

    keyword_docs = []

    for doc in all_docs:
        if query.lower() in doc:
            keyword_docs.append(doc)

    #Combine Docs
    combined_docs = []

    for doc in vector_docs + keyword_docs:
        combined_docs.append(doc)
        


    # return "\n".join(combined_docs)


    #Re-Rank
    pairs = [ (query, doc) for doc in combined_docs]

    scores = reranker.predict(pairs)

    ranked_docs = [
      doc  for _, doc in sorted( zip(scores, combined_docs) ,reverse = True)
    ]

    return "\n\n".join(ranked_docs[:5])



tools = [hybrid_search_pdf]

llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=300,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

llm = llm.bind_tools(tools)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI Assitant, Alwasy repsond like an Aw Moment when answering"),
        MessagesPlaceholder(variable_name = "agent_scratchpad"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

tool_calling_agent = create_tool_calling_agent(
    tools = tools,
    llm =llm,
    prompt = prompt_template
)

agent_executor = AgentExecutor(
    agent = tool_calling_agent,
    tools =  tools,
    verbose=True
)


agent_executor_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_by_id,
    input_messages_key="input",
    history_messages_key="history"
)


while True:
    input_question = input("What is your qustion?")
    if input_question.lower() == 'exit':
        break

    response = agent_executor_with_memory.invoke(
        {"input": input_question},
        config={"configurable": {"session_id": "user_1"}}
    )
    print("Response:", response.get("output"))