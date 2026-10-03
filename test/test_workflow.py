"""Workflow tests use a fake runner, so no LLM calls are made."""
import json
import types

import pytest

from src.agents import build_agents
from src.workflow import InvalidTransitionError, LeoTutorWorkflow, Phase, WorkflowError

QUIZ = {
    "topic": "x", "difficulty": "beginner",
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
ANSWERS = {"q1": "4", "q2": "False", "q3": "combine"}


def make_workflow(scores):
    state = {"eval": 0}

    def raw(text):
        return types.SimpleNamespace(raw=text, pydantic=None, json_dict=None)

    def runner(agent, task, label):
        if label == "Coordinator":
            return raw('{"topic": "Addition", "learner_level": "beginner", "clear": true, "notes": ""}')
        if label == "Explainer":
            return raw(("RETEACH " if "re-teach attempt" in task.description else "STD ") + "explanation " * 10)
        if label == "Quiz Master":
            return raw(json.dumps(QUIZ))
        score = scores[state["eval"]]
        state["eval"] += 1
        fb = [{"question_id": q, "is_correct": score >= 1, "score": score, "feedback": "ok"} for q in ("q1", "q2", "q3")]
        # the fake LLM always claims 99%: the workflow must recompute the real score
        return raw(json.dumps({"overall_score_pct": 99, "per_question": fb, "summary": "s", "recommendation": "proceed"}))

    return LeoTutorWorkflow(agents=build_agents(), runner=runner)


def run_to_answers(wf):
    wf.approve_explanation()
    wf.approve_quiz()


def test_happy_path():
    wf = make_workflow([1.0])
    assert wf.start("Teach me addition") == Phase.EXPLANATION_REVIEW
    run_to_answers(wf)
    assert wf.submit_answers(ANSWERS) == Phase.COMPLETE
    assert not wf.loop_exhausted


def test_low_score_triggers_reteach_then_passes():
    wf = make_workflow([0.2, 1.0])
    wf.start("addition")
    run_to_answers(wf)
    assert wf.submit_answers({"q1": "1"}) == Phase.EXPLANATION_REVIEW
    assert wf.state.reteach_count == 1 and wf.state.explanation.startswith("RETEACH")
    assert wf.attempts[0].evaluation.overall_score_pct == 20.0  # corrected from the LLM's claimed 99
    run_to_answers(wf)
    assert wf.submit_answers(ANSWERS) == Phase.COMPLETE


def test_max_reteach_loops_enforced():
    wf = make_workflow([0.1, 0.1, 0.1])
    wf.start("addition")
    for _ in range(3):
        run_to_answers(wf)
        phase = wf.submit_answers({"q1": "1"})
    assert phase == Phase.COMPLETE and wf.loop_exhausted and wf.state.reteach_count == 2


def test_invalid_transition_and_blank_answers():
    wf = make_workflow([1.0])
    wf.start("addition")
    with pytest.raises(InvalidTransitionError):
        wf.submit_answers(ANSWERS)
    run_to_answers(wf)
    with pytest.raises(WorkflowError):
        wf.submit_answers({"q1": "  "})
    assert wf.phase == Phase.AWAITING_ANSWERS
