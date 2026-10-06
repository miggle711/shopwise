from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse, get_client

from app.config import settings

# General-purpose model used for conversation summarization (see
# conversations.py's maybe_summarise). The product tool-calling LLM lives
# in app/graph.py as a native LangGraph node (product_agent_node), not
# here — this module no longer runs an agent loop of its own.
_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0.7,
)

Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)

langfuse_client = get_client()
