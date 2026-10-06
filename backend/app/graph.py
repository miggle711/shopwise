# backend/app/graph.py
import logging
import os
import sys
from typing import Any, Dict, Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langfuse import get_client
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import PRODUCT_TOOLS
from cart_tools import CART_TOOLS
from order_tools import ORDER_TOOLS
from app.config import settings

logger = logging.getLogger(__name__)

Intent = Literal["product_details", "small_talk", "sensitive_topic", "clarify"]
ALLOWED_INTENTS = {"product_details", "small_talk", "sensitive_topic", "clarify"}

MAX_TOOL_ITERATIONS = 5

ECOMMERCE_SYSTEM_PROMPT = """You are a helpful and professional ecommerce customer service assistant for our online store.

Your responsibilities:
- Help customers find products by answering questions about our catalog
- Use the available tools to search and filter products when customers ask
- Provide information about product specifications, pricing, and availability

Tool usage guidelines:
- Use semantic_search for vague or descriptive queries like "something cozy for winter" or "gift for a fitness lover"
- Use query_products for exact filters like "electronics under $50" or "books with rating above 4.5"
- When filtering by category with query_products, call list_categories first to get exact category names
- You can combine both tools — semantic_search to find relevant products, then describe them with price/rating details
- Use view_order_history when a customer asks about past orders or order/payment status — do not use it for their current cart

Response guidelines:
- Always be polite, professional, and empathetic
- When describing products, include price, rating, and number of reviews
- If you don't know something about our specific products or policies, suggest they contact support"""

ALL_PRODUCT_TOOLS = PRODUCT_TOOLS + CART_TOOLS + ORDER_TOOLS
_product_tool_map = {tool.name: tool for tool in ALL_PRODUCT_TOOLS}

_product_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0.7,
)
_tool_enabled_product_llm = _product_llm.bind_tools(ALL_PRODUCT_TOOLS)


class GraphState(TypedDict, total=False):
    input: str
    chat_history: list[BaseMessage]
    session_id: str
    intent: Intent
    response: str
    error: bool
    tool_calls: list[dict[str, Any]]
    product_messages: list[BaseMessage]
    product_iterations: int


class IntentClassification(BaseModel):
    intent: Intent = Field(
        description="The best matching intent label for the user's message.",
    )


_INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the user's message into exactly one ecommerce routing intent.

Intents:
- product_details: shopping intent, including product search, recommendations, comparisons, prices, availability, product attributes, or follow-up product questions.
- small_talk: greetings, thanks, farewells, or casual conversation unrelated to shopping.
- sensitive_topic: self-harm, violence, abuse, threats, illegal wrongdoing, or safety-sensitive content.
- clarify: unclear messages that cannot be routed using the current message or chat history.

Rules:
- Use chat history to resolve follow-ups like "what about one in blue?"
- Prefer product_details when the user is asking about buying, comparing, finding, or choosing products.
- Prefer sensitive_topic whenever safety risk is present.
- Use clarify only when no other intent clearly fits.

Examples:
- "Show me waterproof jackets under $100" -> product_details
- "What about one in blue?" after discussing jackets -> product_details
- "Can you compare laptops for college?" -> product_details
- "Hey, how are you?" -> small_talk
- "Thanks, that's all" -> small_talk
- "I want to hurt someone" -> sensitive_topic
- "asdf qwerty" -> clarify

Return only the best intent label.""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ]
)

_intent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.google_api_key,
    temperature=0,
)

_intent_classifier = _INTENT_CLASSIFIER_PROMPT | _intent_llm.with_structured_output(IntentClassification)

langfuse_client = get_client()

def _start_graph_span(name: str, state: GraphState):
    return langfuse_client.start_as_current_span(
        name=name,
        input={
            "input": state.get("input", ""),
            "chat_history_length": len(state.get("chat_history", [])),
        },
        metadata={"component": "langgraph"},
    )

async def classify_intent(state: GraphState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    with _start_graph_span("graph.classify_intent", state) as span:
        try:
            result = await _intent_classifier.ainvoke(
                {
                    "input": state.get("input", ""),
                    "chat_history": state.get("chat_history", []),
                },
                config=config,
            )
        except Exception as exc:
            span.update(
                level="ERROR",
                status_message=str(exc),
                output={"intent": "clarify", "error": True},
            )
            logger.exception("Intent classification failed")
            # Routes to clarify_node like a genuine ambiguous-message case
            # (#57 will give this its own dedicated error path), but error=True
            # lets callers (app/routes/chat.py) tell a real failure apart from
            # an ordinary clarify response instead of the two looking
            # identical to the caller (#77).
            return {"intent": "clarify", "error": True}

        intent = getattr(result, "intent", "clarify")
        if intent not in ALLOWED_INTENTS:
            intent = "clarify"

        span.update(output={"intent": intent})
        return {"intent": intent}


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def _init_product_messages(state: GraphState) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=ECOMMERCE_SYSTEM_PROMPT)]
    messages.extend(state.get("chat_history", []))
    user_input = state.get("input", "")
    if user_input:
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=user_input))
    return messages


async def product_agent_node(state: GraphState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    """Calls the tool-bound LLM once per visit. LangGraph's own cycle
    (see route_from_product_agent / product_tools_node) replaces the
    manual `for _ in range(MAX_TOOL_ITERATIONS)` loop this used to run
    inside app/agent.py's AgentExecutorAdapter.
    """
    with _start_graph_span("graph.product_agent_node", state) as span:
        messages = state.get("product_messages") or _init_product_messages(state)
        iterations = state.get("product_iterations", 0) + 1

        ai_message = await _tool_enabled_product_llm.ainvoke(messages, config=config)
        messages = messages + [ai_message]

        response = _message_text(ai_message)
        span.update(output={
            "response": response,
            "has_tool_calls": bool(ai_message.tool_calls),
            "iteration": iterations,
        })

        return {
            "product_messages": messages,
            "product_iterations": iterations,
            "response": response or state.get("response", ""),
        }


async def product_tools_node(state: GraphState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    with _start_graph_span("graph.product_tools_node", state) as span:
        messages = list(state.get("product_messages", []))
        last_message = messages[-1] if messages else None
        tool_calls_log = list(state.get("tool_calls", []))

        for tool_call in getattr(last_message, "tool_calls", []) or []:
            tool_name = tool_call["name"]
            tool = _product_tool_map.get(tool_name)

            if tool is None:
                tool_result = f"Tool '{tool_name}' is not available."
            else:
                # session_id is a trusted value injected by this node, never
                # an LLM-fillable tool argument — tools that need it (cart
                # tools) declare so via a marker attribute, not by exposing
                # session_id in their args_schema.
                call_args = tool_call.get("args", {})
                if getattr(tool, "_needs_session_id", False):
                    call_args = {**call_args, "session_id": state.get("session_id")}
                tool_result = await tool.ainvoke(call_args, config=config)

            tool_calls_log.append({
                "tool": tool_name,
                "args": tool_call.get("args", {}),
                "result": str(tool_result),
            })

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

        span.update(output={"tool_calls": tool_calls_log})
        return {"product_messages": messages, "tool_calls": tool_calls_log}


def route_from_product_agent(state: GraphState) -> str:
    messages = state.get("product_messages", [])
    last_message = messages[-1] if messages else None
    has_pending_tool_calls = bool(getattr(last_message, "tool_calls", None))

    if has_pending_tool_calls and state.get("product_iterations", 0) < MAX_TOOL_ITERATIONS:
        return "product_tools_node"

    return "product_finalize_node"


def product_finalize_node(state: GraphState) -> Dict[str, Any]:
    """Applies the same fallback the old AgentExecutorAdapter loop used
    when the model produces no final text (e.g. it stops after a tool
    call without a closing text turn, or MAX_TOOL_ITERATIONS is hit).
    """
    if state.get("response"):
        return {}
    return {"response": "I apologize, but I'm having trouble generating a response at the moment."}


def small_talk_node(state: GraphState) -> Dict[str, Any]:
    with _start_graph_span("graph.small_talk_node", state) as span:
        response = "response from graph - small talk node placeholder"
        span.update(output={"response": response})
        return {"response": response}


def sensitive_node(state: GraphState) -> Dict[str, Any]:
    with _start_graph_span("graph.sensitive_node", state) as span:
        response = "response from graph - sensitive topic node placeholder"
        span.update(output={"response": response})
        return {"response": response}


async def clarify_node(state: GraphState) -> Dict[str, Any]:
    with _start_graph_span("graph.clarify_node", state) as span:
        response = "response from graph - clarify node placeholder"
        span.update(output={"response": response})
        return {"response": response}


def route_from_intent(state: GraphState) -> str:
    intent = state.get("intent", "clarify")

    if intent == "product_details":
        route = "product_agent_node"
    elif intent == "small_talk":
        route = "small_talk_node"
    elif intent == "sensitive_topic":
        route = "sensitive_node"
    else:
        route = "clarify_node"

    with langfuse_client.start_as_current_span(
        name="graph.route_from_intent",
        input={"intent": intent},
        output={"route": route},
        metadata={"component": "langgraph"},
    ):
        pass

    return route


def build_chat_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("product_agent_node", product_agent_node)
    workflow.add_node("product_tools_node", product_tools_node)
    workflow.add_node("product_finalize_node", product_finalize_node)
    workflow.add_node("small_talk_node", small_talk_node)
    workflow.add_node("sensitive_node", sensitive_node)
    workflow.add_node("clarify_node", clarify_node)

    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_from_intent,
    )

    workflow.add_conditional_edges(
        "product_agent_node",
        route_from_product_agent,
    )
    workflow.add_edge("product_tools_node", "product_agent_node")
    workflow.add_edge("product_finalize_node", END)

    workflow.add_edge("small_talk_node", END)
    workflow.add_edge("sensitive_node", END)
    workflow.add_edge("clarify_node", END)

    return workflow.compile()


chat_graph = build_chat_graph()
