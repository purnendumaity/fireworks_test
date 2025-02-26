import os
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
#from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import FireworksEmbeddings
from langchain_fireworks import FireworksEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_fireworks import Fireworks
#need python-docx package
#from docx import Document

# Set Fireworks API Key
os.environ["FIREWORKS_API_KEY"] = "fw_3ZGdB2sB1mLLbegqQf6vTGSa"

# Initialize FastAPI app
app = FastAPI()
# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

#inputdoc = Document("test.docx")

# Load & process documents
def load_documents():
    loader = UnstructuredWordDocumentLoader("test.docx")  # Replace with actual file path
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embedding = FireworksEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore.as_retriever()

retriever = load_documents()
llm = Fireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")
rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# WebSocket for chatbot communication
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        query = str(data)
        #print("Data type:", type(data))
        #print("Data content:", data)
        print(f"User Question: {data}")
        #response = rag_chain.invoke(query)
        response_dict = rag_chain.invoke(query)  # Returns a dictionary
        print("Response Dict:", response_dict)  # Debugging
        if isinstance(response_dict, dict) and "result" in response_dict:
            response = response_dict["result"]  # Extract the answer
        else:
            response = str(response_dict)  # Ensure response is a string
        #print("Answer content:", response)
        print(f"Bot Answer: {response}")
        await websocket.send_text(response)

# Serve chatbot page
#@app.get("/favicon.ico", include_in_schema=False)
#async def favicon():
#    return {"message": "Favicon not set"}
@app.get("/")
async def serve_chat():
    return StaticFiles(directory="static", html=True)


#uvicorn app:app --reload --host 0.0.0.0 --port 8000
#http://localhost:8000/static/index.html
