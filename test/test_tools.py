import json

from src.tools import check_topic_clarity, grade_objective_answers


def _call(tool_obj, **kwargs):
    """CrewAI's @tool wraps the function; call the raw function if exposed, else .run()."""
    for attr in ("func", "run"):
        fn = getattr(tool_obj, attr, None)
        if callable(fn):
            return fn(**kwargs)
    return tool_obj(**kwargs)


def test_check_topic_clarity():
    assert json.loads(_call(check_topic_clarity, topic="stuff"))["clear"] is False
    assert json.loads(_call(check_topic_clarity, topic=""))["clear"] is False
    assert json.loads(_call(check_topic_clarity, topic="Quantum Physics"))["clear"] is True


def test_grade_objective_answers():
    quiz = {
        "topic": "Math",
        "questions": [
            {"id": f"q{i}", "question_type": "multiple_choice", "question": "What is 2+2?",
             "options": ["2", "3", "4", "5"], "correct_answer": "4",
             "explanation": "2+2=4 always.", "concept_tag": "addition", "points": 1}
            for i in (1, 2, 3)
        ],
    }
    result = json.loads(
        _call(grade_objective_answers, quiz_json=json.dumps(quiz),
              answers_json=json.dumps({"q1": "4", "q2": " 4 ", "q3": "3"}))
    )
    assert result["objective_earned"] == 2 and result["objective_possible"] == 3


def test_grade_objective_answers_handles_bad_input():
    result = json.loads(_call(grade_objective_answers, quiz_json="not json", answers_json="{}"))
    assert "error" in result
