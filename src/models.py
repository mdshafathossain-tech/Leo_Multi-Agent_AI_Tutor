"""
Pydantic data models for Leo.

Three groups of models live here:

1. Quiz Master structured output   -> ``Quiz`` / ``QuizQuestion``
2. Evaluator structured output     -> ``EvaluationResult`` / ``QuestionFeedback``
3. Shared session state (handoffs) -> ``TutorState`` / ``HandoffEvent``

``Quiz`` is passed to CrewAI as ``output_pydantic`` so the Quiz Master's answer is
validated against this schema. Validators enforce internal consistency (e.g. an
MCQ must have 4 options and the correct answer must be one of them).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class QuestionType(str, Enum):
    MCQ = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


# ---------------------------------------------------------------------------
# 1. Quiz Master output
# ---------------------------------------------------------------------------
class QuizQuestion(BaseModel):
    """A single practice question."""

    id: str = Field(..., pattern=r"^q\d+$", description="Unique id such as 'q1', 'q2'.")
    question_type: QuestionType
    question: str = Field(..., min_length=10, description="The question text shown to the student.")
    options: List[str] = Field(
        default_factory=list,
        description="4 options for multiple_choice; ['True','False'] for true_false; empty for short_answer.",
    )
    correct_answer: str = Field(
        ...,
        description="Exact text of the correct option, or a concise model answer for short_answer.",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="For short_answer: the ideas a good answer must contain (used by the Evaluator).",
    )
    explanation: str = Field(..., min_length=10, description="Why the correct answer is correct.")
    concept_tag: str = Field(..., description="Sub-concept tested; enables targeted re-teaching.")
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    points: int = Field(default=1, ge=1, le=10)

    @field_validator("question", "correct_answer", "explanation", "concept_tag")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _check_consistency(self) -> "QuizQuestion":
        if self.question_type == QuestionType.MCQ:
            if len(self.options) != 4 or len(set(self.options)) != 4:
                raise ValueError("multiple_choice questions need exactly 4 distinct options")
            if self.correct_answer not in self.options:
                raise ValueError("correct_answer must exactly match one of the options")
        elif self.question_type == QuestionType.TRUE_FALSE:
            if not self.options:
                self.options = ["True", "False"]
            if self.options != ["True", "False"]:
                raise ValueError("true_false options must be ['True', 'False']")
            if self.correct_answer not in self.options:
                raise ValueError("correct_answer must be 'True' or 'False'")
        else:  # SHORT_ANSWER
            if self.options:
                raise ValueError("short_answer questions must not define options")
            if not self.key_points:
                raise ValueError("short_answer questions need at least one key_point")
        return self


class Quiz(BaseModel):
    """Full quiz produced by the Quiz Master (used as ``output_pydantic``)."""

    topic: str
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    questions: List[QuizQuestion] = Field(..., min_length=3, max_length=10)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Quiz":
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return self

    @property
    def total_points(self) -> int:
        return sum(q.points for q in self.questions)

    def student_view(self) -> List[Dict[str, Any]]:
        """Quiz payload for the UI: hides answers, explanations and key points."""
        hidden = {"correct_answer", "explanation", "key_points"}
        return [q.model_dump(mode="json", exclude=hidden) for q in self.questions]


# ---------------------------------------------------------------------------
# 2. Evaluator output
# ---------------------------------------------------------------------------
class QuestionFeedback(BaseModel):
    question_id: str
    is_correct: bool
    score: float = Field(..., ge=0.0, le=1.0, description="Fraction of the question's points earned.")
    feedback: str = Field(..., description="Constructive, specific feedback for the student.")
    misconception: Optional[str] = Field(default=None, description="Suspected misunderstanding, if any.")


class EvaluationResult(BaseModel):
    """Evaluator verdict, including the routing decision for the feedback loop."""

    overall_score_pct: float = Field(..., ge=0.0, le=100.0)
    per_question: List[QuestionFeedback]
    strengths: List[str] = Field(default_factory=list)
    weak_concepts: List[str] = Field(default_factory=list, description="concept_tags the student struggled with.")
    summary: str
    recommendation: Literal["proceed", "reteach"]
    reteach_focus: Optional[str] = Field(
        default=None, description="Instructions for the Explainer when recommendation == 'reteach'."
    )

    @model_validator(mode="after")
    def _reteach_needs_focus(self) -> "EvaluationResult":
        if self.recommendation == "reteach" and not (self.reteach_focus or self.weak_concepts):
            raise ValueError("reteach requires reteach_focus or weak_concepts")
        return self


# ---------------------------------------------------------------------------
# 3. Shared session state / explicit handoffs
# ---------------------------------------------------------------------------
class HandoffEvent(BaseModel):
    """One agent-to-agent handoff; rendered as a timeline in the Streamlit UI."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    from_agent: str
    to_agent: str
    reason: str
    payload_keys: List[str] = Field(default_factory=list, description="State fields passed along.")


class TutorState(BaseModel):
    """Single source of truth passed between agents (topic -> explanation -> quiz -> evaluation)."""

    topic: str = ""
    learner_level: Difficulty = Difficulty.BEGINNER
    explanation: Optional[str] = None
    explanation_approved: bool = False  # HITL gate 1
    human_notes: Optional[str] = None  # student edits/requests at a HITL gate
    quiz: Optional[Quiz] = None
    quiz_approved: bool = False  # HITL gate 2
    student_answers: Dict[str, str] = Field(default_factory=dict)
    evaluation: Optional[EvaluationResult] = None
    reteach_count: int = 0
    handoffs: List[HandoffEvent] = Field(default_factory=list)

    def log_handoff(self, from_agent: str, to_agent: str, reason: str, *payload_keys: str) -> HandoffEvent:
        event = HandoffEvent(
            from_agent=from_agent, to_agent=to_agent, reason=reason, payload_keys=list(payload_keys)
        )
        self.handoffs.append(event)
        return event

    def reset_for_reteach(self) -> None:
        """Clear quiz-related fields before the Explainer re-teaches; keep topic and evaluation."""
        self.reteach_count += 1
        self.explanation_approved = False
        self.quiz = None
        self.quiz_approved = False
        self.student_answers = {}
