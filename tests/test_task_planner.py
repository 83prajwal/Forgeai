import json

import pytest

from agent.task_planner import _extract_json_collection, parse_plan_steps


def test_planner_parser_preserves_nested_fenced_json_with_prose():
    payload = {"steps": ["Build a map: {name: 'task'}", "Verify nested values"]}
    response = f"Plan follows:\n```json\n{json.dumps(payload)}\n```\nDone."

    assert _extract_json_collection(response) == payload
    assert parse_plan_steps(response) == payload["steps"]


@pytest.mark.parametrize("response", ["", "not JSON", "```json\n{\"steps\":[\n```"])
def test_planner_parser_rejects_empty_malformed_and_truncated_output(response):
    with pytest.raises(ValueError):
        _extract_json_collection(response)
