"""
Leo workflow orchestrator.

The workflow is a small, explicit state machine. Every public method runs the
agents needed for one step and then *returns control to the caller* at each
human-in-the-loop (HITL) gate. This suits Streamlit (which re-runs the script on
every interaction) and works equally well from a console loop.

    start() ─► Coordinator ─► Explainer ─► [GATE 1: EXPLANATION_REVIEW]
        revise_explanation(notes) ─► Explainer again ─► GATE 1
        approve_explanation(notes) ─► Quiz Master ─► [GATE 2: QUIZ_REVIEW]
            regenerate_quiz(notes) ─► Quiz Master again ─► GATE 2
            approve_quiz() ─► AWAITING_ANSWERS
                submit_answers(answers) ─► Evaluator ─┬─ pass ───────────► COMPLETE
                                                      ├─ low score, loops left ─► Explainer
                                                      │      (re-teach) ─► GATE 1 (feedback loop)
                                                      └─ low score, no loops left ─► COMPLETE

Design guarantees
-----------------
* Every agent-to-agent (and agent-to-student) transfer is recorded with
  ``state.log_handoff(...)``.
* Routing is enforced in code: the LLM's "recommendation" is advisory; the
  threshold and loop cap from settings decide.
* A failed step never corrupts state: the phase stays where it was and a
  ``WorkflowError`` is raised so the UI can offer a retry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from crewai import Agent, Crew, Process, Task

from .agents import LeoAgents, build_agents
from .config import Settings, get_settings
from .models import Difficulty, EvaluationResult, Quiz, TutorState
from .tasks import (
    OutputParseError,
    RecoveredOutput,
    TaskBuildError,
    build_coordinator_task,
    build_evaluator_task,
    build_explainer_task,
    build_quiz_task,
    output_text,
    parse_intake,
    parse_structured,
    recover_failed_generation,
)

logger = logging.getLogger("leo.workflow")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
class Phase(str, Enum):
    INTAKE = "intake"
    NEEDS_CLARIFICATION = "needs_clarification"
    EXPLANATION_REVIEW = "explanation_review"  # HITL gate 1
    QUIZ_REVIEW = "quiz_review"  # HITL gate 2
    AWAITING_ANSWERS = "awaiting_answers"
    COMPLETE = "complete"


class WorkflowError(RuntimeError):
    """A step failed (LLM error, unparseable output, missing state). State is left consistent."""


class InvalidTransitionError(WorkflowError):
    """A method was called in a phase where it is not allowed."""


@dataclass
class WorkflowEvent:
    """Real-time log entry; feed these to the Streamlit log panel."""

    agent: str
    kind: str  # start | step | done | handoff | warning | error | info
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AttemptRecord:
    """Snapshot of one quiz attempt (kept because re-teaching clears the live quiz)."""

    attempt_number: int
    quiz: Quiz
    answers: Dict[str, str]
    evaluation: EvaluationResult


EventCallback = Callable[[WorkflowEvent], None]
Runner = Callable[[Agent, Task, str], Any]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
class LeoTutorWorkflow:
    MAX_ATTEMPTS_PER_TASK = 2  # structured tasks: 1 retry with a corrective hint
    MAX_INPUT_CHARS = 2000

    def __init__(
        self,
        agents: Optional[LeoAgents] = None,
        settings: Optional[Settings] = None,
        *,
        on_event: Optional[EventCallback] = None,
        quiz_size: int = 5,
        use_crew_memory: bool = False,
        runner: Optional[Runner] = None,
    ) -> None:
        """
        agents          : inject pre-built agents (else built from env config).
        on_event        : callback receiving ``WorkflowEvent``s for live UI logs.
        quiz_size       : questions per quiz (clamped to 3-10 by the task builder).
        use_crew_memory : enable CrewAI Crew-level memory. Off by default because it
                          needs an embeddings provider (often OpenAI) even when your
                          chat model is a different provider. Explicit ``TutorState``
                          handoffs already carry all required context.
        runner          : override how a task is executed (used for testing).
        """
        self.settings = settings or get_settings()
        self.agents = agents or build_agents(self.settings)
        self.on_event = on_event
        self.quiz_size = quiz_size
        self.use_crew_memory = use_crew_memory
        self._runner: Runner = runner or self._default_runner

        self.state = TutorState()
        self.phase = Phase.INTAKE
        self.clarification: Optional[str] = None  # set when phase == NEEDS_CLARIFICATION
        self.attempts: List[AttemptRecord] = []
        self.loop_exhausted = False  # True if we finished below threshold with no loops left
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ public API
    def start(self, student_input: str, learner_level: Optional[Difficulty] = None) -> Phase:
        """Coordinator validates the request; on success the Explainer runs up to GATE 1."""
        self._require_phase(Phase.INTAKE, Phase.NEEDS_CLARIFICATION, action="start")

        text = (student_input or "").strip()
        if not text:
            return self._ask_clarification("What would you like to learn today?")
        if len(text) > self.MAX_INPUT_CHARS:
            text = text[: self.MAX_INPUT_CHARS]
            self._emit("Coordinator", "warning", f"Input truncated to {self.MAX_INPUT_CHARS} characters.")

        self.state.log_handoff("Student", "Coordinator", "New study request", "student_input")
        self._emit("Coordinator", "handoff", "Student → Coordinator: new study request")

        raw = self._execute(
            "Coordinator",
            self.agents.coordinator,
            lambda hint: build_coordinator_task(
                self.agents.coordinator, text, requested_level=learner_level, retry_hint=hint
            ),
        )
        decision = parse_intake(output_text(raw))
        if not decision.clear:
            return self._ask_clarification(decision.clarifying_question)

        self.clarification = None
        self.state.topic = decision.topic
        self.state.learner_level = learner_level or decision.learner_level
        self.state.log_handoff(
            "Coordinator", "Explainer", "Topic validated and level set", "topic", "learner_level"
        )
        self._emit("Coordinator", "handoff", f"Coordinator → Explainer: '{decision.topic}' ({self.state.learner_level.value})")
        self._explain()
        return self.phase

    def revise_explanation(self, notes: str) -> Phase:
        """GATE 1: student asks for a different explanation; Explainer reruns with the notes."""
        self._require_phase(Phase.EXPLANATION_REVIEW, action="revise_explanation")
        notes = (notes or "").strip()
        if not notes:
            raise WorkflowError("Please describe what should change in the explanation.")
        self.state.human_notes = notes
        self.state.log_handoff("Student", "Explainer", "HITL gate 1: revision requested", "human_notes")
        self._emit("Explainer", "handoff", "Student → Explainer: revision requested")
        self._explain()  # consumes (clears) human_notes on success
        return self.phase

    def approve_explanation(self, notes: Optional[str] = None) -> Phase:
        """GATE 1 approved. Optional ``notes`` become focus areas for the quiz."""
        self._require_phase(Phase.EXPLANATION_REVIEW, action="approve_explanation")
        if not (self.state.explanation or "").strip():
            raise WorkflowError("No explanation in state to approve.")
        self.state.explanation_approved = True
        self.state.human_notes = (notes or "").strip() or None
        self.state.log_handoff(
            "Student", "Quiz Master", "HITL gate 1: explanation approved", "explanation", "human_notes"
        )
        self._emit("Quiz Master", "handoff", "Student → Quiz Master: explanation approved")
        self._generate_quiz()
        return self.phase

    def regenerate_quiz(self, notes: Optional[str] = None) -> Phase:
        """GATE 2: student wants a different quiz; Quiz Master reruns."""
        self._require_phase(Phase.QUIZ_REVIEW, action="regenerate_quiz")
        if notes and notes.strip():
            self.state.human_notes = notes.strip()
        self.state.log_handoff("Student", "Quiz Master", "HITL gate 2: new quiz requested", "human_notes")
        self._emit("Quiz Master", "handoff", "Student → Quiz Master: new quiz requested")
        self._generate_quiz()
        return self.phase

    def approve_quiz(self) -> Phase:
        """GATE 2 approved: the student can now answer the quiz."""
        self._require_phase(Phase.QUIZ_REVIEW, action="approve_quiz")
        if self.state.quiz is None:
            raise WorkflowError("No quiz in state to approve.")
        self.state.quiz_approved = True
        self.phase = Phase.AWAITING_ANSWERS
        self._emit("Quiz Master", "info", "Quiz approved; waiting for the student's answers.")
        return self.phase

    def submit_answers(self, answers: Dict[str, str]) -> Phase:
        """
        Evaluator grades the answers, then routing is enforced:
        pass -> COMPLETE; low score with loops left -> re-teach (back to GATE 1);
        low score with no loops left -> COMPLETE (``loop_exhausted`` = True).
        """
        self._require_phase(Phase.AWAITING_ANSWERS, action="submit_answers")
        quiz = self.state.quiz
        if quiz is None:
            raise WorkflowError("No quiz in state; cannot evaluate.")

        normalized = {q.id: str((answers or {}).get(q.id, "")).strip() for q in quiz.questions}
        unanswered = [qid for qid, a in normalized.items() if not a]
        if len(unanswered) == len(normalized):
            raise WorkflowError("No answers provided. Please answer at least one question.")
        if unanswered:
            self._emit("Evaluator", "warning", f"Unanswered questions (scored 0): {', '.join(unanswered)}")

        self.state.student_answers = normalized
        self.state.log_handoff("Student", "Evaluator", "Answers submitted", "quiz", "student_answers")
        self._emit("Evaluator", "handoff", "Student → Evaluator: answers submitted")

        evaluation = self._execute(
            "Evaluator",
            self.agents.evaluator,
            lambda hint: build_evaluator_task(
                self.agents.evaluator, self.state, pass_threshold=self.settings.pass_threshold, retry_hint=hint
            ),
            EvaluationResult,
        )
        evaluation = self._reconcile_score(evaluation, quiz)

        below = evaluation.overall_score_pct < self.settings.pass_threshold
        loops_left = self.state.reteach_count < self.settings.max_reteach_loops
        self._emit(
            "Evaluator",
            "done",
            f"Score {evaluation.overall_score_pct:.0f}% (threshold {self.settings.pass_threshold:g}%)",
        )

        # Keep a permanent record: reset_for_reteach() clears the live quiz/answers.
        self.attempts.append(
            AttemptRecord(len(self.attempts) + 1, quiz, dict(normalized), evaluation)
        )

        if below and loops_left:
            self._start_reteach(evaluation, quiz)
        else:
            evaluation = evaluation.model_copy(update={"recommendation": "proceed"})
            self.attempts[-1].evaluation = evaluation
            self.state.evaluation = evaluation
            self.loop_exhausted = below
            self.phase = Phase.COMPLETE
            reason = (
                f"Max re-teach loops ({self.settings.max_reteach_loops}) reached; finishing"
                if below
                else "Passed; session complete"
            )
            self.state.log_handoff("Evaluator", "Student", reason, "evaluation")
            self._emit("Evaluator", "handoff", f"Evaluator → Student: {reason}")
        return self.phase

    def reset(self) -> None:
        """Start a brand-new session (keeps agents and settings)."""
        self.state = TutorState()
        self.phase = Phase.INTAKE
        self.clarification = None
        self.attempts = []
        self.loop_exhausted = False
        self.last_error = None

    # ------------------------------------------------------------------ helpers for UI
    @property
    def can_reteach(self) -> bool:
        return self.state.reteach_count < self.settings.max_reteach_loops

    def state_json(self) -> str:
        """Serialised state (handy for debugging/export)."""
        return self.state.model_dump_json(indent=2)

    # ------------------------------------------------------------------ internal steps
    def _explain(self) -> None:
        """Run the Explainer (standard or re-teach) and stop at GATE 1."""
        raw = self._execute(
            "Explainer",
            self.agents.explainer,
            lambda hint: build_explainer_task(self.agents.explainer, self.state, retry_hint=hint),
        )
        explanation = output_text(raw).strip()
        if len(explanation) < 50:
            raise WorkflowError("The Explainer returned an empty or too-short explanation. Please retry.")

        self.state.explanation = explanation
        self.state.explanation_approved = False
        self.state.human_notes = None  # notes are consumed by this run
        self.phase = Phase.EXPLANATION_REVIEW
        self.state.log_handoff(
            "Explainer", "Student", "HITL gate 1: explanation ready for review", "explanation"
        )
        self._emit("Explainer", "handoff", "Explainer → Student: explanation ready (HITL gate 1)")

    def _generate_quiz(self) -> None:
        """Run the Quiz Master and stop at GATE 2."""
        quiz = self._execute(
            "Quiz Master",
            self.agents.quiz_master,
            lambda hint: build_quiz_task(
                self.agents.quiz_master, self.state, num_questions=self.quiz_size, retry_hint=hint
            ),
            Quiz,
        )
        self.state.quiz = quiz.model_copy(update={"topic": self.state.topic})
        self.state.quiz_approved = False
        self.state.student_answers = {}
        self.phase = Phase.QUIZ_REVIEW
        self.state.log_handoff("Quiz Master", "Student", "HITL gate 2: quiz ready for review", "quiz")
        self._emit(
            "Quiz Master", "handoff", f"Quiz Master → Student: {len(quiz.questions)} questions ready (HITL gate 2)"
        )

    def _start_reteach(self, evaluation: EvaluationResult, quiz: Quiz) -> None:
        """Feedback loop: route weak performance from the Evaluator back to the Explainer."""
        weak = evaluation.weak_concepts or self._derive_weak_concepts(evaluation, quiz)
        focus = evaluation.reteach_focus or f"Re-explain these concepts: {', '.join(weak)}."
        evaluation = evaluation.model_copy(
            update={"recommendation": "reteach", "weak_concepts": weak, "reteach_focus": focus}
        )
        self.attempts[-1].evaluation = evaluation
        self.state.evaluation = evaluation

        self.state.reset_for_reteach()  # increments reteach_count, clears quiz + answers
        self.state.human_notes = None
        n, cap = self.state.reteach_count, self.settings.max_reteach_loops
        reason = f"Score {evaluation.overall_score_pct:.0f}% below threshold; re-teach {n}/{cap}"
        self.state.log_handoff("Evaluator", "Explainer", reason, "evaluation", "weak_concepts", "reteach_focus")
        self._emit("Explainer", "handoff", f"Evaluator → Explainer: {reason}")

        # Until the new explanation exists, sit safely at GATE 1 showing the old one.
        self.phase = Phase.EXPLANATION_REVIEW
        try:
            self._explain()
        except WorkflowError:
            self._emit(
                "Explainer",
                "error",
                "Re-teach failed; showing the previous explanation. Use revise_explanation() to retry.",
            )
            raise

    # ------------------------------------------------------------------ evaluation safeguards
    def _reconcile_score(self, evaluation: EvaluationResult, quiz: Quiz) -> EvaluationResult:
        """Recompute the score from per-question results; LLM arithmetic is not trusted."""
        by_id = {fb.question_id: fb for fb in evaluation.per_question}
        points = {q.id: q.points for q in quiz.questions}
        if not set(points).issubset(by_id):
            self._emit("Evaluator", "warning", "Evaluator did not grade every question; using its overall score.")
            return evaluation
        total = sum(points.values())
        computed = round(sum(by_id[qid].score * pts for qid, pts in points.items()) / total * 100, 1)
        if abs(computed - evaluation.overall_score_pct) > 1.0:
            self._emit(
                "Evaluator",
                "warning",
                f"Corrected score {evaluation.overall_score_pct:.0f}% → {computed:.0f}% from per-question results.",
            )
            return evaluation.model_copy(update={"overall_score_pct": computed})
        return evaluation

    @staticmethod
    def _derive_weak_concepts(evaluation: EvaluationResult, quiz: Quiz) -> List[str]:
        tags = {q.id: q.concept_tag for q in quiz.questions}
        weak = [tags[fb.question_id] for fb in evaluation.per_question if fb.score < 0.5 and fb.question_id in tags]
        return list(dict.fromkeys(weak)) or [quiz.topic]  # de-duplicate, keep order

    # ------------------------------------------------------------------ execution plumbing
    def _execute(
        self,
        label: str,
        agent: Agent,
        build_task: Callable[[str], Task],
        model_cls: Optional[type] = None,
    ) -> Any:
        """
        Build and run one task. Every task gets one retry with a corrective hint if the LLM
        call fails or (for structured tasks) the output doesn't validate. When a provider
        rejects a JSON reply that the model tried to send as a tool call, the intended reply
        is recovered from the error instead of failing (see ``recover_failed_generation``).
        Returns the parsed model, or the raw CrewAI output for free-text tasks.
        """
        attempts = self.MAX_ATTEMPTS_PER_TASK
        expects_json = model_cls is not None or label == "Coordinator"
        hint = ""
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                task = build_task(hint)
            except TaskBuildError as exc:
                self._fail(label, str(exc), exc)

            self._emit(label, "start", f"Running (attempt {attempt}/{attempts})")
            try:
                try:
                    output = self._runner(agent, task, label)
                except Exception as run_exc:
                    recovered = recover_failed_generation(str(run_exc)) if expects_json else None
                    if recovered is None:
                        raise
                    self._emit(label, "warning", "Recovered the reply from a rejected tool call (provider quirk).")
                    output = RecoveredOutput(raw=recovered)
                if model_cls is None:
                    self._emit(label, "done", "Task finished")
                    return output
                parsed = parse_structured(output, model_cls)
                self._emit(label, "done", f"Produced valid {model_cls.__name__}")
                return parsed
            except Exception as exc:  # LLM/network errors and OutputParseError alike
                last_exc = exc
                if attempt < attempts:
                    self._emit(label, "warning", f"Attempt {attempt} failed: {str(exc)[:200]}. Retrying once.")
                    if model_cls is not None:
                        hint = (
                            f"Your previous attempt failed ({str(exc)[:300]}). "
                            "Return ONLY valid JSON that matches the required schema, with no extra text."
                        )
                    else:
                        hint = (
                            f"Your previous attempt failed ({str(exc)[:300]}). Write your final answer "
                            "directly as plain text; never call a tool (for example one named 'json') to deliver it."
                        )
                    continue
                break

        self._fail(label, f"{label} failed: {last_exc}", last_exc)

    def _default_runner(self, agent: Agent, task: Task, label: str) -> Any:
        """Run a single task in its own Crew so each handoff is explicit and observable."""

        def _on_step(step: Any) -> None:
            text = getattr(step, "thought", None) or getattr(step, "text", None) or str(step)
            self._emit(label, "step", " ".join(str(text).split())[:300])

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.settings.verbose,
            memory=self.use_crew_memory,
            step_callback=_on_step,
        )
        return crew.kickoff()

    def _ask_clarification(self, question: str) -> Phase:
        self.clarification = question
        self.phase = Phase.NEEDS_CLARIFICATION
        self.state.log_handoff("Coordinator", "Student", "Request unclear; clarification needed")
        self._emit("Coordinator", "handoff", f"Coordinator → Student: {question}")
        return self.phase

    def _require_phase(self, *allowed: Phase, action: str) -> None:
        if self.phase not in allowed:
            names = ", ".join(p.value for p in allowed)
            raise InvalidTransitionError(
                f"Cannot {action} while in phase '{self.phase.value}' (allowed: {names})."
            )

    def _fail(self, label: str, message: str, exc: Optional[BaseException]) -> None:
        self.last_error = message
        self._emit(label, "error", message)
        raise WorkflowError(message) from exc

    def _emit(self, agent: str, kind: str, message: str) -> None:
        logger.info("[%s] %s: %s", agent, kind, message)
        if self.on_event is not None:
            try:
                self.on_event(WorkflowEvent(agent=agent, kind=kind, message=message))
            except Exception:  # a broken UI callback must never break the tutoring flow
                logger.exception("on_event callback failed")


# ---------------------------------------------------------------------------
# Console runner (python -m src.workflow) - handy for testing before the UI exists
# ---------------------------------------------------------------------------
def run_cli() -> None:
    wf = LeoTutorWorkflow(on_event=lambda e: print(f"  [{e.agent}] {e.kind}: {e.message}"))
    text = input("What would you like to learn? ").strip()

    while wf.phase != Phase.COMPLETE:
        try:
            if wf.phase in (Phase.INTAKE, Phase.NEEDS_CLARIFICATION):
                if wf.phase == Phase.NEEDS_CLARIFICATION:
                    text = input(f"\nLeo: {wf.clarification}\n> ").strip()
                wf.start(text)

            elif wf.phase == Phase.EXPLANATION_REVIEW:
                print(f"\n--- Explanation ---\n{wf.state.explanation}\n")
                choice = input("[a]pprove / [r]evise? ").strip().lower()
                if choice.startswith("r"):
                    wf.revise_explanation(input("What should change? "))
                else:
                    wf.approve_explanation(input("Quiz focus areas (optional): "))

            elif wf.phase == Phase.QUIZ_REVIEW:
                for q in wf.state.quiz.student_view():  # type: ignore[union-attr]
                    print(f"{q['id']}: {q['question']}  {q['options'] or ''}")
                if input("\n[a]pprove / [n]ew quiz? ").strip().lower().startswith("n"):
                    wf.regenerate_quiz(input("Any guidance? "))
                else:
                    wf.approve_quiz()

            elif wf.phase == Phase.AWAITING_ANSWERS:
                answers = {}
                for q in wf.state.quiz.student_view():  # type: ignore[union-attr]
                    print(f"\n{q['id']}: {q['question']}")
                    for i, opt in enumerate(q["options"], 1):
                        print(f"   {i}. {opt}")
                    raw = input("Your answer (option number or text): ").strip()
                    if q["options"] and raw.isdigit() and 1 <= int(raw) <= len(q["options"]):
                        raw = q["options"][int(raw) - 1]  # map "3" -> text of option 3
                    answers[q["id"]] = raw
                wf.submit_answers(answers)
        except WorkflowError as exc:
            print(f"\nLeo hit a problem: {exc}")
            if input("Retry? [y/n] ").strip().lower() != "y":
                return

    ev = wf.state.evaluation
    if ev:
        print(f"\nFinal score: {ev.overall_score_pct:.0f}%\n{ev.summary}")
        if wf.loop_exhausted:
            print("(Max re-teach loops reached; consider reviewing the weak concepts on your own.)")


if __name__ == "__main__":
    run_cli()
