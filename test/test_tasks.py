from src.models import Difficulty
from src.tasks import OutputParseError, extract_json, parse_intake

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
