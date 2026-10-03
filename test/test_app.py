"""
UI tests for app.py using Streamlit's AppTest harness.

No LLM is called: ``LeoTutorWorkflow._default_runner`` is replaced by a fake that returns
canned replies, so these tests exercise the real Streamlit script, the real workflow and the
real session-state handling (HITL gates, quiz form, feedback loop).
"""

import json
import types
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.workflow import LeoTutorWorkflow, Phase  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")

QUIZ = {
    "topic": "x",
    "difficulty": "beginner",
    "questions": [
        {"id": "q1", "question_type": "multiple_choice", "question": "What is 2+2 really?",
         "options": ["1", "2", "3", "4"], "correct_answer": "4",
         "explanation": "Basic addition works.", "concept_tag": "add"},
        {"id": "q2", "question_type": "true_false", "question": "The sky is always green.",
         "correct_answer": "False", "explanation": "It is usually blue.", "concept_tag": "sky"},
        {"id": "q3", "question_type": "short_answer", "question": "Explain addition briefly.",
         "correct_answer": "Combining", "key_points": ["combine"],
         "explanation": "Addition combines values.", "concept_tag": "add"},
    ],
}


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the CrewAI runner; returns a list so tests can choose the evaluator scores."""
    scores = [1.0]
    counter = {"eval": 0}

    def raw(text):
        return types.SimpleNamespace(raw=text, pydantic=None, json_dict=None)

    def fake_runner(self, agent, task, label):
        if label == "Coordinator":
            return raw('{"topic": "Addition", "learner_level": "beginner", "clear": true, "notes": ""}')
        if label == "Explainer":
            return raw(("RETEACH " if "re-teach attempt" in task.description else "STD ") + "explanation " * 10)
        if label == "Quiz Master":
            return raw(json.dumps(QUIZ))
        score = scores[min(counter["eval"], len(scores) - 1)]
        counter["eval"] += 1
        fb = [{"question_id": q, "is_correct": score >= 1, "score": score, "feedback": "ok"}
              for q in ("q1", "q2", "q3")]
        return raw(json.dumps({"overall_score_pct": 99, "per_question": fb,
                               "summary": "summary", "recommendation": "proceed"}))

    monkeypatch.setattr(LeoTutorWorkflow, "_default_runner", fake_runner)
    return scores


def _click(at, text):
    button = next(b for b in at.button if text in b.label)
    button.click().run()
    return at


def _new_app():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def _to_gate1(at):
    at.text_area[0].input("Teach me addition")
    _click(at, "Start learning")
    assert not at.exception
    assert at.session_state["wf"].phase == Phase.EXPLANATION_REVIEW


def _to_answers(at):
    _click(at, "Approve & build quiz")
    assert at.session_state["wf"].phase == Phase.QUIZ_REVIEW
    _click(at, "Approve")
    assert at.session_state["wf"].phase == Phase.AWAITING_ANSWERS


def _answer(at, q1, q2, q3):
    at.radio[0].set_value(q1)
    at.radio[1].set_value(q2)
    at.text_area[0].input(q3)
    _click(at, "Submit answers")
    assert not at.exception


def test_happy_path_through_both_hitl_gates(fake_llm):
    at = _new_app()
    _to_gate1(at)
    assert any("Review" in str(m.value) or "Your turn" in str(m.value) for m in at.markdown)
    _to_answers(at)
    _answer(at, "4", "False", "combine")
    wf = at.session_state["wf"]
    assert wf.phase == Phase.COMPLETE
    assert any("passed" in s.value for s in at.success)


def test_revise_at_gate1_requires_feedback_then_reruns_explainer(fake_llm):
    at = _new_app()
    _to_gate1(at)
    _click(at, "Revise explanation")  # empty feedback box
    assert any("what to change" in w.value for w in at.warning)
    assert at.session_state["wf"].phase == Phase.EXPLANATION_REVIEW
    at.text_area[0].input("Simpler please")
    _click(at, "Revise explanation")
    wf = at.session_state["wf"]
    assert wf.phase == Phase.EXPLANATION_REVIEW
    assert any(h.reason.startswith("HITL gate 1: revision") for h in wf.state.handoffs)


def test_low_score_shows_feedback_loop_banner(fake_llm):
    fake_llm[:] = [0.1, 1.0]
    at = _new_app()
    _to_gate1(at)
    _to_answers(at)
    _answer(at, "1", "True", "no idea")
    wf = at.session_state["wf"]
    assert wf.phase == Phase.EXPLANATION_REVIEW and wf.state.reteach_count == 1
    assert any("Feedback loop" in w.value for w in at.warning)
    # second round passes
    _to_answers(at)
    _answer(at, "4", "False", "combine")
    assert at.session_state["wf"].phase == Phase.COMPLETE


def test_blank_submission_is_rejected_without_calling_evaluator(fake_llm):
    at = _new_app()
    _to_gate1(at)
    _to_answers(at)
    _click(at, "Submit answers")
    assert at.session_state["wf"].phase == Phase.AWAITING_ANSWERS
    assert any("at least one" in w.value for w in at.warning)
