import os
from langfuse.callback import CallbackHandler
from langchain_core.messages import HumanMessage
from graph.graph_builder import graph

langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")

# Initialize Langfuse CallbackHandler for LangGraph/Langchain (tracing)
langfuse_handler = CallbackHandler(
    public_key=langfuse_public_key,
    secret_key=langfuse_secret_key,
    host="http://localhost:3000"
)

question_text = "Generami il drawio."  # Replace with the actual question you want to ask
file_name = "ark.png"  # Replace with the actual file name you want to process

messages = HumanMessage(content=question_text + " Path: files/" + file_name)
answer = graph.invoke(input={"messages": messages}, config={"callbacks": [langfuse_handler]})