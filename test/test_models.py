import pytest
from pydantic import ValidationError

from src.models import Quiz, QuizQuestion, QuestionType, TutorState


def _mcq(qid: str) -> QuizQuestion:
    return QuizQuestion(
        id=qid,
        question_type=QuestionType.MCQ,
        question="What is the capital of France?",
        options=["Paris", "London", "Berlin", "Madrid"],
        correct_answer="Paris",
        explanation="Paris is the capital of France.",
        concept_tag="geography",
    )


def test_valid_quiz_model():
    quiz = Quiz(topic="Geography", questions=[_mcq("q1"), _mcq("q2"), _mcq("q3")])
    assert quiz.total_points == 3
    assert len(quiz.student_view()) == 3
    assert "correct_answer" not in quiz.student_view()[0]  # answers stay hidden from the student


def test_duplicate_question_ids_rejected():
    with pytest.raises(ValidationError):
        Quiz(topic="Geography", questions=[_mcq("q1"), _mcq("q1"), _mcq("q1")])


def test_invalid_mcq_options():
    with pytest.raises(ValidationError):
        QuizQuestion(
            id="q1",
            question_type=QuestionType.MCQ,
            question="What is 2+2?",
            options=["3", "4"],  # fewer than 4 options must fail
            correct_answer="4",
            explanation="Math fact.",
            concept_tag="math",
        )


def test_correct_answer_must_be_an_option():
    with pytest.raises(ValidationError):
        QuizQuestion(
            id="q1",
            question_type=QuestionType.MCQ,
            question="What is 2+2?",
            options=["1", "2", "3", "5"],
            correct_answer="4",
            explanation="Math fact.",
            concept_tag="math",
        )


def test_reset_for_reteach_clears_quiz_but_keeps_explanation():
    state = TutorState(topic="t", explanation="e" * 60)
    state.quiz = Quiz(topic="t", questions=[_mcq("q1"), _mcq("q2"), _mcq("q3")])
    state.student_answers = {"q1": "Paris"}
    state.reset_for_reteach()
    assert state.reteach_count == 1
    assert state.quiz is None and state.student_answers == {}
    assert state.explanation
