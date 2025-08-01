from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import sys
import uvicorn
import logging
from data_loader import HospitalDataLoader
from chat import chatbot_instance
from memory import InMemoryUserMemoryStore
from chat import chat

console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log", encoding='utf-8'),
        logging.StreamHandler(console_stream)
    ]
)
logger = logging.getLogger(__name__)

retriever = chatbot_instance.retriever
# Initialize data loader
data_loader = HospitalDataLoader()
# Custom in-memory user memory store
user_memory_store = InMemoryUserMemoryStore()

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("indextag.html", {"request": request})

@app.post("/refresh_data")
async def refresh_data_endpoint():
    logger.info("Hospital data refresh request received.")
    global data_loader
    data_loader = HospitalDataLoader()
    retriever.refresh_indexs()
    logger.info("Hospital data and retrieval models refreshed successfully.")
    return {"message": "Hospital data and retrieval models refreshed successfully."}

@app.get("/api/metadata-tags")
async def get_metadata_tags():
    tags = data_loader.get_all_metadata_tags()
    return {"tags": tags}

@app.get("/api/tag-counts")
async def get_metadata_tag_counts():
    return {"tag_counts": data_loader.get_metadata_tag_counts()}

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: Request, chat_data: ChatInput, x_user_id: str = Header(None)):
    user_message = chat_data.message.strip()
    if not user_message:
        return JSONResponse(status_code=400, content={"error": "No message provided"})

    user_id = x_user_id or request.client.host
    response = chat(user_message, user_id)
    logger.info(f"Chat response generated for user {user_id}")
    print(response)
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)