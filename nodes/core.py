from states.state import AgentState
import os
# Import the load_dotenv function from the dotenv library
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.object_detection_tools import detect_objects_in_image as object_detection_tool
from tools.object_detection_tools import select_latest_image_versions as select_latest_image_versions_tool
from tools.object_detection_tools import verify_bounding_boxes as verify_bounding_boxes_tool
from tools.object_detection_tools import adjust_bounding_boxes_in_image as adjust_bounding_boxes_in_image_tool
from tools.drawio_tools import generate_drawio_from_image_and_objects as drawio_tool
from langfuse.callback import CallbackHandler
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated

load_dotenv()

# Read your API key from the environment variable or set it manually
api_key = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro-preview-06-05")
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "128"))
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")

# Initialize Langfuse CallbackHandler for LangGraph/Langchain (tracing)
langfuse_handler = CallbackHandler(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    host="http://localhost:3000"
)

chat = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    temperature=0,
    max_retries=2,
    google_api_key=api_key,
    thinking_budget=GEMINI_THINKING_BUDGET
)

object_detection_agent = create_react_agent(
    model=chat,
    tools=[object_detection_tool, select_latest_image_versions_tool, verify_bounding_boxes_tool, adjust_bounding_boxes_in_image_tool],
    name="object_detection_agent",
    prompt="You are an expert into extracting objects from images." \
    "You need to extract 3rd party custom pics (not present in default drawio library) from the images of the user." \
    "Which pics to extract, is something you need to decide with your tools and will not be provided by the user." \
    "These pics will be then used to build a drawio diagram." \
    "Once you have extracted the bounding boxes, verify if the result is correct with your tools" \
    "You have access to tools for object detection: one for generating the first detection, another for corrections. " \
    "The resulting bounding boxes will be used to generate a drawio diagram. " \
    "You need to return the bounding boxes labels, in case there are spaces insert underscore." \
    "Your labels will be used by another agent to read the images and use them to build a drawio diagram." \
    "After 3 iterations with your tools, stop using the tool and provide the answer, even if partial." \
    "Before returning the result, you need to call the select latest image versions tool and return the list provided by the tool. " \
)

drawio_generator_agent = create_react_agent(
    model=chat,
    tools=[drawio_tool],
    name="drawio_generator_agent",
    prompt="You are an expert into generating drawio diagrams. Your goal is to generate a drawio based on the image in input.")

# Create supervisor workflow
workflow = create_supervisor(
    [object_detection_agent, drawio_generator_agent],
    model=chat,
    prompt=(
        "You are a team supervisor managing a process that involves object detection and diagram generation. "
        "Your goal is to generate a drawio file following the steps: " \
        "- Extract objects from the original image that user provides in local path (use object_detection_agent)" \
        "- Generate a drawio using the images extracted at the previous step (use drawio_generator_agent)" \
    )
)

# Compile and run
graph = workflow.compile()

