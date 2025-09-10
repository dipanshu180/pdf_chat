# main.py - FastAPI Backend (Render-ready, Python 3.10, no SQLAlchemy)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import tempfile
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables (expects OPENAI_API_KEY in .env or Render Dashboard)
load_dotenv()

app = FastAPI(title="PDF Chat API", description="Chat with your PDF documents using AI")

# Allow all origins for frontend testing/deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for frontend
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store conversation sessions in memory
sessions: dict = {}

class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str

# ---- Helper Functions ---- #

def get_pdf_text(pdf_files: List[UploadFile]) -> str:
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.file.read())
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, "rb") as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    extracted = page.extract_text() or ""
                    text += extracted + "\n"
        finally:
            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {tmp_path}: {e}")

    return text.strip()

def get_text_chunks(text: str) -> List[str]:
    """Split text into smaller chunks for embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks: List[str]):
    """Create an in-memory FAISS vectorstore."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    """Create a conversational chain with memory."""
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def format_response(answer: str) -> str:
    """
    Organize response: Convert bullet points, headings, etc.
    - Converts `- item` to `<li>`
    - Wraps in <ul>
    - Bold for **text**
    """
    import re
    formatted = answer
    formatted = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", formatted)
    formatted = re.sub(r"^- (.*)$", r"<li>\1</li>", formatted, flags=re.MULTILINE)
    if "<li>" in formatted:
        formatted = f"<ul>{formatted}</ul>"
    formatted = formatted.replace("\n", "<br>")
    return formatted

# ---- API Endpoints ---- #

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend HTML."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>PDF Chat API is Running</h1>")

@app.post("/upload-pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDFs, returning a session_id."""
    try:
        session_id = str(uuid.uuid4())
        raw_text = get_pdf_text(files)
        if not raw_text:
            raise HTTPException(status_code=400, detail="No text found in uploaded PDFs")

        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        conversation_chain = get_conversation_chain(vectorstore)

        sessions[session_id] = {
            'conversation': conversation_chain,
            'chat_history': []
        }

        return {
            "message": f"Processed {len(files)} PDF(s)",
            "session_id": session_id,
            "chunks_created": len(text_chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask questions about the uploaded PDFs."""
    try:
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found. Upload PDFs first.")
        session = sessions[request.session_id]
        conversation = session['conversation']
        response = conversation({'question': request.question})
        session['chat_history'] = response['chat_history']

        answer = format_response(response['answer'])

        return ChatResponse(answer=answer, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
