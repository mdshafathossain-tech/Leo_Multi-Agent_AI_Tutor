"""
CrewAI task builders for Leo.

Each ``build_*_task`` function turns the current ``TutorState`` into one CrewAI
``Task``. Handoffs are explicit: everything an agent needs (topic, explanation,
quiz, answers, evaluation) is injected into its prompt from the state object,
so no agent depends on hidden conversational context.

Also contains the parsing helpers that make LLM output safe to consume:
``extract_json``, ``parse_structured`` and ``parse_intake``.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Type, TypeVar

from crewai import Agent, Task
from pydantic import BaseModel, ValidationError

from .agents import (
    COORDINATOR_TASK_TEMPLATE,
    EVALUATOR_TASK_TEMPLATE,
    EXPLAINER_TASK_TEMPLATE,
    QUIZ_TASK_TEMPLATE,
    RETEACH_TASK_TEMPLATE,
)
from .models import Difficulty, EvaluationResult, Quiz, TutorState

T = TypeVar("T", bound=BaseModel)

DEFAULT_CLARIFICATION = (
    "Could you tell me a bit more? Which subject and which concept would you like to learn?"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TaskBuildError(ValueError):
    """Raised when required state is missing to build a task (e.g. no explanation yet)."""


class OutputParseError(ValueError):
    """Raised when an LLM reply cannot be parsed/validated into the expected structure."""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def extract_json(text: Optional[str]) -> Any:
    """
    Pull the first JSON value out of an LLM reply.
    Handles ```json fences``` and prose before/after the JSON.
    """
    if not text or not text.strip():
        raise OutputParseError("Empty reply; expected JSON.")

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
            return value
        except json.JSONDecodeError:
            continue
    raise OutputParseError("No valid JSON found in reply.")


@dataclass
class RecoveredOutput:
    """Minimal stand-in for a CrewAI output when a reply is recovered from an API error."""

    raw: str
    pydantic: Any = None
    json_dict: Any = None


def recover_failed_generation(error_text: Optional[str]) -> Optional[str]:
    """
    Recover the model's intended JSON reply from a rejected tool call.

    Some providers (e.g. Groq with gpt-oss models) answer HTTP 400 ``tool_use_failed`` when the
    model tries to deliver its final JSON as a call to a non-existent tool named "json". The
    error body still contains the intended reply in ``failed_generation``.

    Returns the reply as a JSON string, or None when nothing safe can be recovered (for example
    when the rejected call targeted a real tool, whose arguments are not a final answer).
    """
    if not error_text or "failed_generation" not in error_text:
        return None

    generation: Any = None
    # 1) The exception text embeds the provider's JSON body as a Python dict repr.
    start, end = error_text.find("{'"), error_text.rfind("}")
    if start != -1 and end > start:
        try:
            body = ast.literal_eval(error_text[start : end + 1])
            err = body.get("error", body) if isinstance(body, dict) else None
            if isinstance(err, dict):
                generation = err.get("failed_generation")
        except (ValueError, SyntaxError):
            generation = None
    # 2) Fallback: pull the quoted value out of the raw text.
    if generation is None:
        match = re.search(r"failed_generation['\"]?\s*:\s*['\"](.*)['\"]\s*\}", error_text, re.DOTALL)
        if match:
            generation = match.group(1).replace("\\n", "\n").replace("\\'", "'")
    if not isinstance(generation, str):
        return None

    try:
        data = extract_json(generation)
    except OutputParseError:
        return None
    if not isinstance(data, dict):
        return None
    if "arguments" in data:  # {"name": "json", "arguments": {...}}
        if str(data.get("name", "json")).strip().lower() != "json":
            return None  # a real tool was called; its arguments are not the final answer
        args = data["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        data = args
    if not isinstance(data, dict) or not data:
        return None
    return json.dumps(data, ensure_ascii=False)


def output_text(output: Any) -> str:
    """Best-effort plain-text view of a CrewOutput / TaskOutput / str."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    raw = getattr(output, "raw", None)
    return raw if isinstance(raw, str) else str(output)


def parse_structured(output: Any, model_cls: Type[T]) -> T:
    """
    Convert a CrewAI output into ``model_cls``.
    Order: native ``output_pydantic`` -> ``json_dict`` -> JSON extracted from raw text.
    """
    native = getattr(output, "pydantic", None)
    if isinstance(native, model_cls):
        return native

    errors: List[str] = []
    json_dict = getattr(output, "json_dict", None)
    if isinstance(json_dict, dict):
        try:
            return model_cls.model_validate(json_dict)
        except ValidationError as exc:
            errors.append(_short_validation_error(exc))

    try:
        payload = extract_json(output_text(output))
        return model_cls.model_validate(payload)
    except OutputParseError as exc:
        errors.append(str(exc))
    except ValidationError as exc:
        errors.append(_short_validation_error(exc))

    raise OutputParseError(f"Could not build {model_cls.__name__}: " + " | ".join(errors))


def _short_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Coordinator intake
# ---------------------------------------------------------------------------
@dataclass
class IntakeDecision:
    clear: bool
    topic: str = ""
    learner_level: Difficulty = Difficulty.BEGINNER
    notes: str = ""
    clarifying_question: str = ""


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return default if value is None else bool(value)


def parse_intake(raw_text: str) -> IntakeDecision:
    """
    Interpret the Coordinator's reply. Per the prompt it returns JSON when the
    request is clear, and a plain-text clarifying question when it is not.
    """
    try:
        data = extract_json(raw_text)
    except OutputParseError:
        question = (raw_text or "").strip() or DEFAULT_CLARIFICATION
        return IntakeDecision(clear=False, clarifying_question=question)

    if not isinstance(data, dict):
        return IntakeDecision(clear=False, clarifying_question=DEFAULT_CLARIFICATION)

    topic = str(data.get("topic") or "").strip()
    clear = _as_bool(data.get("clear"), default=bool(topic)) and bool(topic)
    try:
        level = Difficulty(str(data.get("learner_level") or "beginner").strip().lower())
    except ValueError:
        level = Difficulty.BEGINNER
    notes = str(data.get("notes") or "").strip()
    question = "" if clear else str(data.get("clarifying_question") or notes or DEFAULT_CLARIFICATION)
    return IntakeDecision(clear, topic, level, notes, question)


# Compact, valid example of the Quiz JSON. Embedded in the prompt instead of exposing a
# zero-argument "schema" tool, which providers such as Groq reject (empty tool schema).
QUIZ_FORMAT_EXAMPLE = """\
{
  "topic": "<topic>",
  "difficulty": "beginner",
  "questions": [
    {"id": "q1", "question_type": "multiple_choice", "question": "<question text>",
     "options": ["<A>", "<B>", "<C>", "<D>"], "correct_answer": "<exactly one of the options>",
     "key_points": [], "explanation": "<why>", "concept_tag": "<sub-concept>", "difficulty": "beginner", "points": 1},
    {"id": "q2", "question_type": "true_false", "question": "<statement>",
     "options": ["True", "False"], "correct_answer": "True",
     "key_points": [], "explanation": "<why>", "concept_tag": "<sub-concept>", "difficulty": "beginner", "points": 1},
    {"id": "q3", "question_type": "short_answer", "question": "<question text>",
     "options": [], "correct_answer": "<concise model answer>",
     "key_points": ["<idea 1>", "<idea 2>"], "explanation": "<why>", "concept_tag": "<sub-concept>", "difficulty": "beginner", "points": 2}
  ]
}
Rules: ids are q1, q2, ... and unique; multiple_choice has exactly 4 distinct options;
true_false options are exactly ["True", "False"]; short_answer has no options but needs key_points."""


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------
def _with_hint(description: str, retry_hint: str) -> str:
    return f"{description}\n\nIMPORTANT: {retry_hint}" if retry_hint else description


def _bullets(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_coordinator_task(
    agent: Agent,
    student_input: str,
    *,
    requested_level: Optional[Difficulty] = None,
    retry_hint: str = "",
) -> Task:
    """Validate topic clarity and set the learner level."""
    if not student_input or not student_input.strip():
        raise TaskBuildError("student_input is empty.")
    description = COORDINATOR_TASK_TEMPLATE.format(student_input=student_input.strip())
    if requested_level is not None:
        description += f"\nThe student selected the level '{requested_level.value}'; use it."
    return Task(
        description=_with_hint(description, retry_hint),
        expected_output=(
            "Either a JSON object with keys topic, learner_level, clear, notes, "
            "or one short clarifying question in plain text."
        ),
        agent=agent,
    )


def build_explainer_task(agent: Agent, state: TutorState, *, retry_hint: str = "") -> Task:
    """
    Standard explanation, or re-teaching (RETEACH_TASK_TEMPLATE) when the
    feedback loop has fired (reteach_count > 0 and an evaluation exists).
    """
    if not state.topic.strip():
        raise TaskBuildError("Cannot explain: topic is missing from state.")

    is_reteach = state.reteach_count > 0 and state.evaluation is not None and bool(state.explanation)
    if is_reteach:
        ev = state.evaluation
        assert ev is not None  # for type checkers
        description = RETEACH_TASK_TEMPLATE.format(
            score=ev.overall_score_pct,
            topic=state.topic,
            weak_concepts=_bullets(ev.weak_concepts or ["(see evaluator guidance)"]),
            reteach_focus=ev.reteach_focus or "Focus on the weak concepts listed above.",
            previous_explanation=state.explanation,
            attempt=state.reteach_count,
        )
        if state.human_notes:
            description += f"\nStudent's extra requests: {state.human_notes}"
        expected = "A focused Markdown re-explanation of the weak concepts, under 450 words."
    else:
        description = EXPLAINER_TASK_TEMPLATE.format(
            topic=state.topic,
            learner_level=state.learner_level.value,
            human_notes=state.human_notes or "None",
        )
        expected = "A structured Markdown explanation with the six requested sections, under 600 words."

    return Task(description=_with_hint(description, retry_hint), expected_output=expected, agent=agent)


def build_quiz_task(
    agent: Agent,
    state: TutorState,
    *,
    num_questions: int = 5,
    retry_hint: str = "",
) -> Task:
    """Quiz Master task; output is validated against the ``Quiz`` model."""
    if not state.explanation or not state.explanation.strip():
        raise TaskBuildError("Cannot create a quiz: explanation is missing from state.")
    if not state.topic.strip():
        raise TaskBuildError("Cannot create a quiz: topic is missing from state.")

    description = QUIZ_TASK_TEMPLATE.format(
        num_questions=max(3, min(10, int(num_questions))),  # Quiz model allows 3-10
        topic=state.topic,
        difficulty=state.learner_level.value,
        explanation=state.explanation,
        human_notes=state.human_notes or "None",
    )
    description += "\n\nFormat example:\n" + QUIZ_FORMAT_EXAMPLE
    return Task(
        description=_with_hint(description, retry_hint),
        expected_output="A JSON object that validates against the Quiz schema.",
        agent=agent,
        output_pydantic=Quiz,
    )


def build_evaluator_task(
    agent: Agent,
    state: TutorState,
    *,
    pass_threshold: float,
    retry_hint: str = "",
) -> Task:
    """Evaluator task; output is validated against the ``EvaluationResult`` model."""
    if state.quiz is None:
        raise TaskBuildError("Cannot evaluate: quiz is missing from state.")
    if not state.student_answers:
        raise TaskBuildError("Cannot evaluate: no student answers in state.")

    description = EVALUATOR_TASK_TEMPLATE.format(
        quiz_json=state.quiz.model_dump_json(),
        answers_json=json.dumps(state.student_answers, ensure_ascii=False),
        pass_threshold=f"{pass_threshold:g}",
    )
    return Task(
        description=_with_hint(description, retry_hint),
        expected_output="A JSON object that validates against the EvaluationResult schema.",
        agent=agent,
        output_pydantic=EvaluationResult,
    )
