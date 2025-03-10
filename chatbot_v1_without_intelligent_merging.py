#Need to install unstructured
#Need to install Python-docx, openai
import os
import json
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_fireworks import FireworksEmbeddings
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
#from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
#from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_fireworks import Fireworks

# ✅ Set API Keys
os.environ["FIREWORKS_API_KEY"] = "fw_3ZRR1TURKGdkw8HEhrZcNaTV"

# ✅ Initialize FastAPI app
start_time = time.time()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
print(f"🚀 Uvicorn Application Startup: Started at {datetime.now().strftime('%H:%M:%S')}")

# ✅ Configuration for Answer Mode
#ANSWER_MODE = "HYBRID"  # Change to any of 3 mode: 'LLM_ONLY', 'KB_ONLY', 'HYBRID'
#CURRENT_ANSWER_MODE = "KB_ONLY" # Default mode

# ✅ File paths
INDEX_FILE = "./faissdb/faiss_index"
DOCS_FOLDER = "./input"

# ✅ Text Splitter
text_splitter = SentenceTransformersTokenTextSplitter(
    model_name="all-MiniLM-L6-v2",
    chunk_size=600
)

# ✅ Custom Fireworks Embeddings Class (Prevents Malformed Inputs)
class SafeFireworksEmbeddings(FireworksEmbeddings):
    def embed_query(self, query):
        if isinstance(query, dict):
            query = query.get("question", "").strip()

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"⚠️ Fireworks received invalid query: {query}")

        print(f"🔥 Fireworks Embedding Debug: Query -> {query}")
        return super().embed_query(query)

# ✅ Initialize Embeddings
embeddings = SafeFireworksEmbeddings(model_name="nomic-ai/nomic-embed-text-v1")

# ✅ Load Documents
def load_documents():
    """Loads multiple Word documents and splits them into chunks."""
    start = time.time()
    print(f"📄 Loading documents at {datetime.now().strftime('%H:%M:%S')}...")

    doc_paths = [os.path.join(DOCS_FOLDER, f) for f in os.listdir(DOCS_FOLDER) if f.endswith(".docx")]

    if not doc_paths:
        print("⚠️ No documents found in input folder.")
        return []

    documents = []
    for path in doc_paths:
        loader = UnstructuredWordDocumentLoader(path)
        doc_content = loader.load()
        split_docs = text_splitter.split_documents(doc_content)

        for idx, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = idx
            doc.metadata["source"] = path

        documents.extend(split_docs)

    print(f"✅ Documents Loaded in {round(time.time() - start, 2)} sec.")
    return documents

# ✅ Create or Load FAISS Vectorstore
vectorstore = None
def create_or_update_vectorstore(documents):
    """Creates FAISS index if not exists, otherwise updates it."""
    start = time.time()
    print(f"📌 FAISS Indexing Started at {datetime.now().strftime('%H:%M:%S')}...")

    if not documents:
        print("⚠️ No documents to index. Skipping FAISS update.")
        return None

    split_documents = text_splitter.split_documents(documents)

    if os.path.exists(INDEX_FILE):
        print("✅ Loading existing FAISS index...")
        vectorstore = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)

        # Ensure FAISS index dimensions match
        existing_dim = vectorstore.index.d
        new_dim = len(embeddings.embed_query("test"))
        if existing_dim != new_dim:
            print(f"⚠️ FAISS dimension mismatch! Rebuilding index.")
            os.remove(INDEX_FILE)
            return create_or_update_vectorstore(documents)

        # Add new documents if not already indexed
        existing_texts = {doc.page_content for doc in vectorstore.docstore._dict.values()}
        new_documents = [doc for doc in split_documents if doc.page_content not in existing_texts]

        if new_documents:
            print(f"➕ Adding {len(new_documents)} new documents to FAISS index...")
            vectorstore.add_documents(new_documents)
            vectorstore.save_local(INDEX_FILE)

    else:
        print("🆕 Creating new FAISS index...")
        vectorstore = FAISS.from_documents(split_documents, embeddings)
        vectorstore.save_local(INDEX_FILE)

    # ✅ Print FAISS index size
    total_docs = len(vectorstore.docstore._dict)
    print(f"✅ FAISS Index Ready with {total_docs} documents in {round(time.time() - start, 2)} sec.")
    return vectorstore

# ✅ Retrieve FAISS Index (Lazy Load)
def get_faiss_retriever():
    global vectorstore
    if vectorstore is None:
        print("⚠️ FAISS not loaded! Initializing now...")
        documents = load_documents()
        vectorstore = create_or_update_vectorstore(documents)
    return vectorstore.as_retriever(search_kwargs={"k": 10}) if vectorstore else None

# ✅ Define Knowledgebase only answer mode Prompt
knowledge_rag_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant with access to the following reference documents:
{context}

You MUST answer strictly based on the given documents. 
If the answer is not found, respond: "I don’t know based on the provided documents."

Chat history:
{chat_history}

Question: {question}
Answer: """
)

# ✅ Define Hybrid answer mode Prompt
hybrid_rag_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant with access to reference documents:
{context}

Use the documents first, but if they don’t fully answer, then use llm model to give answer.

Chat history:
{chat_history}

Question: {question}
Answer: """
)

# ✅ Initialize Fireworks LLM
llm = Fireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct", max_tokens=900)

# ✅ Define Memory
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
memory = ConversationSummaryMemory(
    llm=llm,  # Pass the LLM instance
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
)

def get_memory(_):
    return {"chat_history": memory.load_memory_variables({}).get("chat_history", [])}
memory_runnable = RunnableLambda(get_memory)

# ✅ Define RAG Workflow
#retrieval_rag_chain_old = (
#    RunnableParallel({
#        "context": get_faiss_retriever(),
#        "question": RunnableLambda(lambda x: x),
#        "chat_history": memory_runnable
#    })
#    | custom_rag_prompt
#    | llm
#)

# ✅ Define RAG Workflow Based on Answer Mode
def get_response_chain_new(answer_mode):
    if answer_mode == "LLM_ONLY":
        return RunnableLambda(lambda query: llm.invoke(query["question"]))  # ✅ Extract string before passing

    elif answer_mode == "KB_ONLY":
        return (
                RunnableParallel({
                    "context": lambda x: get_faiss_retriever().invoke(x),
                    "question": RunnableLambda(lambda x: x),
                    "chat_history": memory_runnable
                })
                | knowledge_rag_prompt
                | llm
        )

    elif answer_mode == "HYBRID":
        def hybrid_context(query):
            retrieved_docs = get_faiss_retriever().invoke(query)
            if not retrieved_docs:
                print("⚠️ No relevant documents found, falling back to general LLM knowledge.")
                return None  # Signal direct LLM call
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        return RunnableLambda(
            lambda query: (
                llm.invoke(query["question"]) if hybrid_context(query["question"]) is None
                else (
                        RunnableParallel({
                            "context": RunnableLambda(lambda q: hybrid_context(q["question"])),
                            "question": RunnableLambda(lambda q: q["question"]),
                            "chat_history": memory_runnable
                        })
                        | hybrid_rag_prompt
                        | llm
                )
            )
        )

    else:
        raise ValueError("Invalid ANSWER_MODE! Choose from 'LLM_ONLY', 'KB_ONLY', or 'HYBRID'")

# ✅ Retrieve Documents
def retrieve_documents(query):
    """Retrieves relevant documents from FAISS."""
    start = time.time()
    print(f"📌 Retrieving Documents for Query: '{query}' at {datetime.now().strftime('%H:%M:%S')}")

    if isinstance(query, dict):
        query = query.get("question", "").strip()

    retriever = get_faiss_retriever()
    retrieved_chunks = retriever.invoke(query) if retriever else []

    print(f"✅ Retrieved {len(retrieved_chunks)} Documents in {round(time.time() - start, 2)} sec.")
    return retrieved_chunks

# ✅ WebSocket API (Timing Logs)
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"bot": "Hello! 👋 Ask me anything from the documents!"})

    while True:
        try:
            # 🔹 Receive data from frontend
            data = await websocket.receive_text()

            # 🔹 Convert JSON string to dictionary (handles question + mode)
            try:
                query_data = json.loads(data)
                question = query_data.get("question", "").strip()  # Get user input (default: empty string)
                selected_answer_mode = query_data.get("mode", "KB_ONLY")  # Get answer mode (default: KB_ONLY)
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON received!")
                await websocket.send_text("⚠️ Error: Invalid request format. Please try again.")
                continue  # Skip this iteration if JSON is invalid

            start = time.time()
            print(f"📌 Received Query: '{question}', Mode: {selected_answer_mode}")

            # ✅ Use the latest selected answer mode for response generation
            response_chain = get_response_chain_new(selected_answer_mode)

            # 🔍 Retrieve relevant documents
            retrieved_docs = retrieve_documents(question)
            context = "\n\n".join(
                doc.page_content for doc in retrieved_docs) if retrieved_docs else "No relevant documents found."

            print(f"📌 Starting LLM Processing for Query: '{question}' at {datetime.now().strftime('%H:%M:%S')}")

            # 🧠 Generate response using selected answer mode
            response_dict = response_chain.invoke({"question": question})

            # ✅ Ensure response is always a string (fixes 'str' object has no attribute 'get' error)
            response = response_dict["result"] \
            if isinstance(response_dict,dict) and "result" in response_dict else str(response_dict)

            print(f"✅ LLM Response Received in {round(time.time() - start, 2)} sec.")
            print(f"📝 Response Preview: {response[:100]}...")

            await websocket.send_text(response)

        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            await websocket.send_text("⚠️ An error occurred while processing your query.")

# ✅ Startup Completion Log
print(f"✅ Uvicorn Ready to Accept Requests in {round(time.time() - start_time, 2)} sec.")
