from src.models import Difficulty
from src.tasks import OutputParseError, extract_json, parse_intake, recover_failed_generation

import pytest


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(OutputParseError):
        extract_json("no json here")


def test_parse_intake_clear_request():
    d = parse_intake('{"topic": "Photosynthesis", "learner_level": "advanced", "clear": true, "notes": ""}')
    assert d.clear and d.topic == "Photosynthesis" and d.learner_level == Difficulty.ADVANCED


def test_parse_intake_plain_text_is_a_clarifying_question():
    d = parse_intake("Which subject do you mean?")
    assert not d.clear and d.clarifying_question == "Which subject do you mean?"


def test_parse_intake_string_false_is_not_clear():
    assert not parse_intake('{"topic": "x", "clear": "false"}').clear


# The exact error the Groq gpt-oss model produced when it tried to "call" a tool named json.
GROQ_ERROR = (
    "Error code: 400 - {'error': {'message': \"Tool call validation failed: tool call validation failed: "
    "attempted to call tool 'json' which was not in request.tools\", 'type': 'invalid_request_error', "
    "'code': 'tool_use_failed', 'failed_generation': '{\"name\": \"json\", \"arguments\": {\\n "
    "\"topic\": \"supply and demand\",\\n \"learner_level\": \"beginner\",\\n \"clear\": true,\\n "
    "\"notes\": \"The request is clear.\"\\n}}'}}"
)


def test_recover_failed_generation_from_json_tool_call():
    import json

    recovered = recover_failed_generation(GROQ_ERROR)
    assert recovered is not None
    data = json.loads(recovered)
    assert data["topic"] == "supply and demand" and data["clear"] is True
    assert parse_intake(recovered).clear


def test_recover_failed_generation_ignores_other_errors_and_real_tools():
    assert recover_failed_generation("Error code: 404 - model_not_found") is None
    assert recover_failed_generation(None) is None
    real_tool = GROQ_ERROR.replace('"name": "json"', '"name": "check_topic_clarity"')
    assert recover_failed_generation(real_tool) is None
