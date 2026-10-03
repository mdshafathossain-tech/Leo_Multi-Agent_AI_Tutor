"""
Tools available to Leo's agents.

* ``check_topic_clarity``    (Coordinator) - cheap heuristic to catch vague/empty topics.
* ``grade_objective_answers``(Evaluator)   - deterministic grading of MCQ / true-false answers,
                                             so scores aren't left to LLM arithmetic.
* ``build_search_tool``      (Explainer)   - optional web search, only if SERPER_API_KEY is set.
"""

from __future__ import annotations

import json
from typing import List

from crewai.tools import BaseTool, tool

from .config import get_settings
from .models import Quiz, QuestionType

_VAGUE_TOPICS = {"stuff", "things", "anything", "something", "help", "idk", "everything", "school"}


@tool("check_topic_clarity")
def check_topic_clarity(topic: str) -> str:
    """Check whether a student's topic is specific enough to teach.
    Returns JSON: {"clear": bool, "issue": str | null}."""
    cleaned = (topic or "").strip()
    words = cleaned.lower().split()
    issue = None
    if len(cleaned) < 3:
        issue = "Topic is empty or too short."
    elif len(words) == 1 and words[0] in _VAGUE_TOPICS:
        issue = "Topic is too vague; ask which subject and concept the student wants."
    elif len(cleaned) > 300:
        issue = "Topic is too long; ask the student to summarise it in one sentence."
    return json.dumps({"clear": issue is None, "issue": issue})


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


@tool("grade_objective_answers")
def grade_objective_answers(quiz_json: str, answers_json: str) -> str:
    """Grade multiple_choice and true_false answers deterministically.
    quiz_json: the Quiz as JSON. answers_json: JSON object {question_id: student_answer}.
    Returns JSON with per-question results and points earned on objective questions.
    short_answer questions are flagged 'needs_review' for the Evaluator to judge."""
    try:
        quiz = Quiz.model_validate_json(quiz_json)
        answers = json.loads(answers_json)
    except Exception as exc:  # malformed input from the LLM
        return json.dumps({"error": f"Invalid input: {exc}"})

    results: List[dict] = []
    earned = possible = 0
    for q in quiz.questions:
        given = answers.get(q.id, "")
        if q.question_type == QuestionType.SHORT_ANSWER:
            results.append({"question_id": q.id, "status": "needs_review", "student_answer": given})
            continue
        correct = _norm(given) == _norm(q.correct_answer)
        possible += q.points
        earned += q.points if correct else 0
        results.append(
            {
                "question_id": q.id,
                "status": "correct" if correct else "incorrect",
                "student_answer": given,
                "correct_answer": q.correct_answer,
                "concept_tag": q.concept_tag,
                "points": q.points if correct else 0,
            }
        )
    return json.dumps({"objective_earned": earned, "objective_possible": possible, "results": results})


def build_search_tool() -> List[BaseTool]:
    """Return a web-search tool for the Explainer if configured, else an empty list."""
    if not get_settings().serper_enabled:
        return []
    from crewai_tools import SerperDevTool  # imported lazily; reads SERPER_API_KEY from env

    return [SerperDevTool()]