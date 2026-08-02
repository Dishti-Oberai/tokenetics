from tokenetics.core.cache_hash import stable_prefix_hash
from tokenetics.core.request import ToolSpec

_TOOL_A = ToolSpec(name="get_weather", description="Get the weather", input_schema={"type": "object"})
_TOOL_B = ToolSpec(name="send_email", description="Send an email", input_schema={"type": "object"})


def test_same_system_and_tools_same_hash():
    assert stable_prefix_hash("Sys.", [_TOOL_A]) == stable_prefix_hash("Sys.", [_TOOL_A])


def test_different_system_different_hash():
    assert stable_prefix_hash("Sys A.", [_TOOL_A]) != stable_prefix_hash("Sys B.", [_TOOL_A])


def test_different_tools_different_hash():
    assert stable_prefix_hash("Sys.", [_TOOL_A]) != stable_prefix_hash("Sys.", [_TOOL_B])


def test_tool_order_matters():
    # Caching is prefix-based on literal bytes -- a different tool order
    # is a different literal prefix, so it must hash differently too.
    assert stable_prefix_hash("Sys.", [_TOOL_A, _TOOL_B]) != stable_prefix_hash(
        "Sys.", [_TOOL_B, _TOOL_A]
    )


def test_none_system_no_tools_is_stable():
    assert stable_prefix_hash(None, []) == stable_prefix_hash(None, [])
