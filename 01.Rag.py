from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()  # Load environment variables from .env file
import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


groq_api_key = os.getenv("groq_api")


llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=300,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_docs")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')



# //Load PDF

# Check if chroma_db is empty

async def create_embeddings():

    if len(collection.get()) == 0:
        print("ChromaDB is empty. Loading PDF and creating embeddings...")

        loader = PyPDFLoader(
                file_path = r"C:\Users\sreek\LangChainProjects\metarpin.pdf",
                
        )

        print(len(loader.load()))

        Pages = []

        for index, page in enumerate(loader.load()):
            Pages.append({
                "page_content": page.page_content,
                "metadata": page.metadata,
                "source": index
            })

        print(len(Pages))


        # Splitter
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)

        for page in Pages:
            page_chunks = text_splitter.split_text(page["page_content"])
            print(f"Page {page['source']} has {len(page_chunks)} chunks.")
            for chunk in page_chunks:
                chunks.append({
                    "page_content": chunk,
                    "metadata": page["metadata"],
                    "source": page["source"]
                })

        print(f"Total chunks created: {len(chunks)}")


        texts = [chunk['page_content'] for chunk in chunks]

        chunk_embeddings = embed_model.encode(texts, normalize_embeddings=True)

        collection.add(
            documents=texts,
            embeddings=chunk_embeddings.tolist(),
            metadatas=[chunk["metadata"] for chunk in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
    else:
        print("ChromaDB already has data. Skipping PDF loading and embedding creation.")


async def query_embeddings(query):
    query_embedding = embed_model.encode(query, normalize_embeddings=True)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    print("Top 3 relevant chunks:")
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        print(f"Chunk {i+1}: {doc[:200]}...")  # Print the first 200 characters of each chunk
        context += doc + "\n"

    return context




store = {}

def get_session_history_by_id(id):
    if id not in store:
        store[id] = []
    return store[id]


   
        
async def task():
    await create_embeddings()
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        embedding_context =  await query_embeddings(question)
 
       
        prompt = f"""
                You are a helpful assistant. Answer ONLY using the context below.

                If the answer is not in the context, say "I don't know".

                Context:
                {embedding_context}

                Question:
                {question}

                Answer:
                """

        llm_wi
        
        response = llm.invoke(prompt)
        print("LLM Response:", response.content)




if __name__ == "__main__":
    import asyncio
    asyncio.run(task())