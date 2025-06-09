from states.state import AgentState
import os
# Import the load_dotenv function from the dotenv library
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.object_detection_tools import detect_objects_in_image as object_detection_tool
from tools.drawio_tools import generate_drawio_from_image_and_objects as drawio_tool
from tools.drawio_tools import save_drawio_xml as drawio_saver_tool
from langfuse.callback import CallbackHandler

load_dotenv()

# Read your API key from the environment variable or set it manually
api_key = os.getenv("GEMINI_API_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")

# Initialize Langfuse CallbackHandler for LangGraph/Langchain (tracing)
langfuse_handler = CallbackHandler(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    host="http://localhost:3000"
)

chat = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash-preview-05-20",
    temperature=0,
    max_retries=2,
    google_api_key=api_key,
    thinking_budget= 0
)

tools = [
    object_detection_tool,
    drawio_tool,
]

chat_with_tools = chat.bind_tools(tools)

def assistant(state: AgentState):
    sys_msg = "You are a helpful assistant with access to tools. Your goal is to generate a drawio file following the steps:" \
    "- Extract objects from the original image that user provides" \
    "- Generate a drawio using the images extracted at the previous step"
    return {
        "messages": [chat_with_tools.invoke([sys_msg] + state["messages"])]
    }