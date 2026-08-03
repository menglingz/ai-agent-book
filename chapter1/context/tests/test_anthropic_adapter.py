"""Anthropic protocol adapter: response parsing, NO_TOOL_RESULTS placeholder
tool_result blocks, and NO_REASONING thinking-block filtering.

Uses SimpleNamespace stand-ins for Anthropic SDK response objects rather than
the real SDK types, so these tests run without a network call and stay fast.
"""
from types import SimpleNamespace

from agent import AnthropicAdapter, ContextMode


def _block(block_type, **kwargs):
    data = {"type": block_type, **kwargs}
    return SimpleNamespace(model_dump=lambda: data, **data)


def _response(*blocks):
    return SimpleNamespace(
        content=list(blocks),
        model_dump=lambda: {"content": [b.model_dump() for b in blocks]},
    )


def test_parse_response_extracts_text_tool_use_and_thinking():
    response = _response(
        _block("thinking", thinking="reasoning about the task"),
        _block("text", text="here is my answer"),
        _block("tool_use", id="tu_1", name="calculate", input={"expression": "1+1"}),
    )
    adapter = AnthropicAdapter()
    parsed = adapter.parse_response(response)

    assert parsed.text == "here is my answer"
    assert parsed.reasoning_text == "reasoning about the task"
    assert parsed.tool_calls == [
        {"id": "tu_1", "name": "calculate", "arguments": {"expression": "1+1"}, "parse_error": None}
    ]
    assert parsed.response_dict["content"][1]["type"] == "text"


def test_parse_response_handles_text_only_reply():
    response = _response(_block("text", text="Hi! How can I help you today?"))
    parsed = AnthropicAdapter().parse_response(response)

    assert parsed.text == "Hi! How can I help you today?"
    assert parsed.tool_calls == []
    assert parsed.reasoning_text is None


def test_format_tool_result_message_keeps_block_structure_when_hidden():
    """NO_TOOL_RESULTS must not drop the tool_result block: Anthropic requires
    every tool_use to be followed by a matching tool_result in the very next
    message, or the API returns 400. Only the content is replaced."""
    adapter = AnthropicAdapter()
    visible = adapter.format_tool_result_message("tu_1", "42")
    hidden = adapter.format_tool_result_message("tu_1", "[Tool result hidden due to context mode]")

    for msg in (visible, hidden):
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "tu_1"

    assert visible["content"][0]["content"] == "42"
    assert hidden["content"][0]["content"] == "[Tool result hidden due to context mode]"


def test_format_assistant_message_filters_thinking_block_in_no_reasoning_mode():
    response = _response(
        _block("thinking", thinking="secret reasoning"),
        _block("text", text="answer"),
    )
    adapter = AnthropicAdapter()

    full = adapter.format_assistant_message(response, ContextMode.FULL)
    no_reasoning = adapter.format_assistant_message(response, ContextMode.NO_REASONING)

    assert [b["type"] for b in full["content"]] == ["thinking", "text"]
    assert [b["type"] for b in no_reasoning["content"]] == ["text"]


def test_split_system_extracts_system_role_into_top_level_param():
    adapter = AnthropicAdapter()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hi"},
    ]
    system, rest = adapter._split_system(messages)

    assert system == "You are a helpful assistant."
    assert rest == [{"role": "user", "content": "hi"}]


def test_tools_schema_translates_openai_shape_to_input_schema():
    tools_description = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a math expression",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            },
        }
    ]
    schema = AnthropicAdapter().tools_schema(tools_description)

    assert schema == [
        {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        }
    ]


def test_build_request_moves_system_message_to_top_level():
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "task"},
    ]
    create_kwargs, request_data = AnthropicAdapter().build_request(
        model="auto", messages=messages, tools_description=None,
        temperature=0.3, max_tokens=8192, provider="anthropic", using_openrouter=False,
    )

    assert create_kwargs["system"] == "sys prompt"
    assert create_kwargs["messages"] == [{"role": "user", "content": "task"}]
    assert "tools" not in create_kwargs or create_kwargs.get("tools") is None
    assert request_data["system"] == "sys prompt"
