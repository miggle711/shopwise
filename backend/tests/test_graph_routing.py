# backend/tests/test_graph.py
import os
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Placeholder-string matching (e.g. checking for "test-key") is brittle —
# CI uses a different placeholder ("dummy-key-for-tests") than local dev's
# conftest.py default ("test-key"), and neither actually looks like a real
# key. Real Google API keys start with "AIza"; anything else is a stand-in.
# The primary guard is still `pytest -m "not llm"` in CI — this is a
# secondary safety net for anyone running the full suite locally without
# that flag but without a real key either.
_HAS_REAL_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").startswith("AIza")

LIVE_LLM_INTENT_CASES = [
    ("Do you have any running shoes under $80?", "product_details"),
    ("Can you compare the best laptops for college?", "product_details"),
    ("Hey there, how's your day going?", "small_talk"),
    ("Thanks, that's all I needed today.", "small_talk"),
    ("I want to hurt someone who scammed me.", "sensitive_topic"),
    ("asdf qwerty", "clarify"),
]

# Tests routing
async def test_classify_intent_routes_known_prompts(mocker):
    import app.graph as graph

    mock_classifier = mocker.Mock()
    mock_classifier.ainvoke = AsyncMock(side_effect=[
        graph.IntentClassification(intent="product_details"),
        graph.IntentClassification(intent="small_talk"),
        graph.IntentClassification(intent="sensitive_topic"),
        graph.IntentClassification(intent="product_details"),
        graph.IntentClassification(intent="clarify"),
    ])
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    assert (await graph.classify_intent({"input": "Do you have any running shoes under $80?"}))["intent"] == "product_details"
    assert (await graph.classify_intent({"input": "Hey there, how's your day going?"}))["intent"] == "small_talk"
    assert (await graph.classify_intent({"input": "I want to hurt someone who scammed me."}))["intent"] == "sensitive_topic"
    assert (await graph.classify_intent({"input": "Can you compare the best laptops for college?"}))["intent"] == "product_details"

    history_state = {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }
    assert (await graph.classify_intent(history_state))["intent"] == "clarify"

    assert mock_classifier.ainvoke.call_args_list[4].args[0] == {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }

# Tests fallback behavior
async def test_classify_intent_falls_back_to_clarify(mocker):
    import app.graph as graph

    class InvalidClassification:
        intent = "not_a_real_intent"

    mock_classifier = mocker.Mock()
    mock_classifier.ainvoke = AsyncMock(return_value=InvalidClassification())
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    result = await graph.classify_intent({"input": "asdf qwerty"})

    assert result["intent"] == "clarify"
    # Classifier ran successfully (just returned an unrecognized label) —
    # not an infra failure, so no error flag (contrast with the LLM-errors
    # test below).
    assert "error" not in result

# Tests fallback behavior when LLM errors
async def test_classify_intent_falls_back_to_clarify_when_llm_errors(mocker):
    import app.graph as graph

    mock_classifier = mocker.Mock()
    mock_classifier.ainvoke = AsyncMock(side_effect=RuntimeError("temporary model failure"))
    mocker.patch.object(graph, "_intent_classifier", mock_classifier)

    result = await graph.classify_intent({"input": "I'm looking for a gift but not sure what kind"})

    assert result["intent"] == "clarify"
    # A real classifier failure (e.g. quota exhaustion) must be distinguishable
    # from a genuine ambiguous-message clarify, not silently look like a normal
    # successful reply to the caller (#77).
    assert result["error"] is True


def test_graph_compiles():
    import app.graph as graph

    assert graph.chat_graph is not None


async def test_product_agent_node_returns_text_response_when_no_tool_calls(mocker):
    import app.graph as graph

    ai_message = AIMessage(content="Here are three laptop options under $900.")
    ai_message.tool_calls = []
    mock_llm = mocker.Mock()
    mock_llm.ainvoke = AsyncMock(return_value=ai_message)
    mocker.patch.object(graph, "_tool_enabled_product_llm", mock_llm)

    state = {
        "input": "Show me laptops under $900",
        "chat_history": [HumanMessage(content="I need something for school.")],
    }

    result = await graph.product_agent_node(state)

    assert result["response"] == "Here are three laptop options under $900."
    assert result["product_iterations"] == 1
    assert graph.route_from_product_agent({**state, **result}) == "product_finalize_node"


async def test_product_agent_node_routes_to_tools_node_when_model_calls_a_tool(mocker):
    import app.graph as graph

    ai_message = AIMessage(content="")
    ai_message.tool_calls = [{"name": "semantic_search", "args": {"query": "laptop"}, "id": "call_1"}]
    mock_llm = mocker.Mock()
    mock_llm.ainvoke = AsyncMock(return_value=ai_message)
    mocker.patch.object(graph, "_tool_enabled_product_llm", mock_llm)

    state = {"input": "Show me laptops under $900", "chat_history": []}
    result = await graph.product_agent_node(state)

    assert graph.route_from_product_agent({**state, **result}) == "product_tools_node"


async def test_route_from_product_agent_stops_looping_at_max_iterations(mocker):
    import app.graph as graph

    tool_call_message = AIMessage(content="")
    tool_call_message.tool_calls = [{"name": "semantic_search", "args": {}, "id": "call_1"}]

    state = {
        "product_messages": [tool_call_message],
        "product_iterations": graph.MAX_TOOL_ITERATIONS,
    }

    assert graph.route_from_product_agent(state) == "product_finalize_node"


async def test_product_tools_node_invokes_matching_tool_and_logs_call(mocker):
    import app.graph as graph

    mock_tool = mocker.Mock(spec=["ainvoke"])
    mock_tool.ainvoke = AsyncMock(return_value="3 laptops found")
    mocker.patch.object(graph, "_product_tool_map", {"semantic_search": mock_tool})

    ai_message = AIMessage(content="")
    ai_message.tool_calls = [{"name": "semantic_search", "args": {"query": "laptop"}, "id": "call_1"}]

    state = {"product_messages": [ai_message], "tool_calls": []}
    result = await graph.product_tools_node(state)

    mock_tool.ainvoke.assert_called_once_with({"query": "laptop"}, config=None)
    assert result["tool_calls"] == [
        {"tool": "semantic_search", "args": {"query": "laptop"}, "result": "3 laptops found"}
    ]
    assert isinstance(result["product_messages"][-1], ToolMessage)
    assert result["product_messages"][-1].content == "3 laptops found"


async def test_product_tools_node_injects_session_id_for_marked_tools(mocker):
    import app.graph as graph

    mock_tool = mocker.Mock()
    mock_tool._needs_session_id = True
    mock_tool.ainvoke = AsyncMock(return_value="Cart updated")
    mocker.patch.object(graph, "_product_tool_map", {"add_to_cart": mock_tool})

    ai_message = AIMessage(content="")
    ai_message.tool_calls = [{"name": "add_to_cart", "args": {"product_id": "p1"}, "id": "call_1"}]

    state = {
        "product_messages": [ai_message],
        "tool_calls": [],
        "session_id": "session-123",
    }
    await graph.product_tools_node(state)

    mock_tool.ainvoke.assert_called_once_with(
        {"product_id": "p1", "session_id": "session-123"}, config=None
    )


async def test_product_tools_node_handles_unknown_tool_name(mocker):
    import app.graph as graph

    mocker.patch.object(graph, "_product_tool_map", {})

    ai_message = AIMessage(content="")
    ai_message.tool_calls = [{"name": "not_a_real_tool", "args": {}, "id": "call_1"}]

    result = await graph.product_tools_node({"product_messages": [ai_message], "tool_calls": []})

    assert "not available" in result["tool_calls"][0]["result"]


def test_product_finalize_node_keeps_existing_response():
    import app.graph as graph

    assert graph.product_finalize_node({"response": "Here are some laptops."}) == {}


def test_product_finalize_node_falls_back_when_no_response():
    import app.graph as graph

    assert graph.product_finalize_node({"response": ""}) == {
        "response": "I apologize, but I'm having trouble generating a response at the moment."
    }

# Tests live LLM classification (requires a real GOOGLE_API_KEY)
@pytest.mark.llm
@pytest.mark.skipif(
    not _HAS_REAL_GOOGLE_API_KEY,
    reason="A real GOOGLE_API_KEY is required.",
)
@pytest.mark.parametrize(("prompt", "expected_intent"), LIVE_LLM_INTENT_CASES)
async def test_classify_intent_with_live_llm(prompt, expected_intent):
    import app.graph as graph

    assert (await graph.classify_intent({"input": prompt}))["intent"] == expected_intent

# Tests live LLM classification with chat history (requires a real GOOGLE_API_KEY)
@pytest.mark.llm
@pytest.mark.skipif(
    not _HAS_REAL_GOOGLE_API_KEY,
    reason="A real GOOGLE_API_KEY is required.",
)
async def test_classify_intent_with_live_llm_uses_chat_history():
    import app.graph as graph

    state = {
        "input": "What about one in blue?",
        "chat_history": [HumanMessage(content="Show me waterproof jackets for hiking.")],
    }

    assert (await graph.classify_intent(state))["intent"] == "product_details"
