"""
Leo's four agents (CrewAI).

    Coordinator -> Explainer -> Quiz Master -> Evaluator --(low score)--> Explainer
                       ^ HITL gate                ^ HITL gate

This module only *defines* the agents and the prompt templates their tasks will
use. Tasks, the sequential/feedback-loop flow and HITL gates are wired up in
``src/crew.py`` / ``src/workflow.py`` (Phase 2).

Design notes
------------
* Only the Coordinator may delegate; specialists stay focused on one job.
* Memory: CrewAI memory is toggled by LEO_AGENT_MEMORY. It supplements, but does
  not replace, the explicit ``TutorState`` handoff object in ``models.py``.
* Every task prompt below receives the context it needs via ``str.format`` so
  handoffs are explicit and visible in the UI logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from crewai import Agent

from .config import Settings, build_llm, get_settings
from .tools import (
    build_search_tool,
    check_topic_clarity,
    grade_objective_answers,
)

# ---------------------------------------------------------------------------
# Task prompt templates (formatted with TutorState fields in Phase 2)
# ---------------------------------------------------------------------------
COORDINATOR_TASK_TEMPLATE = """\
A student sent this request: "{student_input}"

1. Use the check_topic_clarity tool on the topic you extract.
2. If the request is unclear, unsafe, or off-topic for studying, do NOT proceed:
   reply with a friendly clarifying question.
3. Otherwise output JSON: {{"topic": str, "learner_level": "beginner|intermediate|advanced",
   "clear": true, "notes": str}}.
Keep the topic to one concise phrase."""

EXPLAINER_TASK_TEMPLATE = """\
Teach the topic "{topic}" to a {learner_level} learner.

Structure your explanation with these sections:
1. Big idea (2-3 sentences, plain language)
2. Key concepts (3-5 bullets, each with a one-line definition)
3. Worked example (step by step)
4. Analogy from everyday life
5. Common mistakes to avoid
6. Quick recap (3 bullets)

Student's extra requests (may be empty): {human_notes}
Keep it under 600 words. Use Markdown. Do not write quiz questions."""

RETEACH_TASK_TEMPLATE = """\
The student scored {score:.0f}% on the quiz for "{topic}" and struggled with:
{weak_concepts}

Evaluator guidance: {reteach_focus}

Previous explanation (do not simply repeat it):
{previous_explanation}

Re-teach ONLY the weak concepts using a different angle: a new analogy, a new
worked example, and a short "why the common misconception is wrong" section.
Use Markdown and stay under 450 words. This is re-teach attempt #{attempt}."""

QUIZ_TASK_TEMPLATE = """\
Create a {num_questions}-question practice quiz on "{topic}" at {difficulty} difficulty,
based ONLY on this explanation:

{explanation}

Rules:
- Mix multiple_choice (exactly 4 options), true_false, and at least one short_answer.
- Give every question a concept_tag so weak areas can be re-taught.
- Provide correct_answer and a clear explanation for every question.
- Student focus areas (may be empty): {human_notes}
Return ONLY a JSON object (no prose, no markdown fences) that follows the format shown below."""

EVALUATOR_TASK_TEMPLATE = """\
Evaluate the student's answers.

Quiz (JSON): {quiz_json}
Student answers (JSON): {answers_json}
Pass threshold: {pass_threshold}%

Steps:
1. Call grade_objective_answers for multiple_choice and true_false questions.
2. Judge each short_answer against its key_points; award partial credit fairly.
3. Write specific, encouraging feedback per question (say what was right first).
4. Compute overall_score_pct from points earned / total points.
5. If overall_score_pct < {pass_threshold}: recommendation = "reteach", list weak_concepts
   and write reteach_focus for the Explainer. Otherwise recommendation = "proceed".
Return ONLY JSON matching the EvaluationResult schema."""


# ---------------------------------------------------------------------------
# Agent container + factory
# ---------------------------------------------------------------------------
@dataclass
class LeoAgents:
    coordinator: Agent
    explainer: Agent
    quiz_master: Agent
    evaluator: Agent

    def as_dict(self) -> Dict[str, Agent]:
        return {
            "Coordinator": self.coordinator,
            "Explainer": self.explainer,
            "Quiz Master": self.quiz_master,
            "Evaluator": self.evaluator,
        }


def build_agents(settings: Settings | None = None) -> LeoAgents:
    """Create all four agents. Raises early if the LLM key is missing."""
    settings = settings or get_settings()

    # Slightly different temperatures per role: creative teaching vs. strict grading.
    llm_coordinator = build_llm(settings, temperature=0.2)
    llm_explainer = build_llm(settings, temperature=0.5)
    llm_quiz = build_llm(settings, temperature=0.4)
    llm_evaluator = build_llm(settings, temperature=0.1)

    common = dict(
        verbose=settings.verbose,
        memory=settings.agent_memory,
        respect_context_window=True,  # auto-summarise if context grows too large
        max_iter=6,  # cap reasoning loops to control cost
    )

    coordinator = Agent(
        role="Leo Coordinator",
        goal=(
            "Understand what the student wants to learn, reject or clarify unclear requests, "
            "and route work to the right specialist with the context they need."
        ),
        backstory=(
            "You are the friendly front desk of Leo, an AI study team. You have years of "
            "experience as a teaching-centre advisor and can turn a vague 'help me with "
            "science' into a precise learning goal. You never guess: when a request is "
            "ambiguous you ask one short clarifying question. You handle errors calmly and "
            "explain what went wrong in plain language."
        ),
        llm=llm_coordinator,
        tools=[check_topic_clarity],
        allow_delegation=True,  # only the Coordinator orchestrates
        **common,
    )

    explainer = Agent(
        role="Leo Explainer",
        goal=(
            "Teach the requested topic clearly and accurately at the student's level, using "
            "structured explanations, analogies and worked examples."
        ),
        backstory=(
            "You are a patient tutor who has taught thousands of students. You believe that "
            "every concept can be understood if it is broken into small steps and linked to "
            "everyday experience. You adapt tone and depth to the learner, and when a student "
            "struggles you try a completely different angle instead of repeating yourself."
        ),
        llm=llm_explainer,
        tools=build_search_tool(),  # empty unless SERPER_API_KEY is configured
        allow_delegation=False,
        **common,
    )

    quiz_master = Agent(
        role="Leo Quiz Master",
        goal=(
            "Generate fair, well-formed practice questions grounded in the explanation, "
            "returned strictly as JSON that validates against the Quiz schema."
        ),
        backstory=(
            "You are an assessment designer who writes exam questions for a living. You "
            "write unambiguous questions, plausible distractors and precise answer keys, and "
            "you tag each question with the concept it tests. You only test what was taught "
            "and you always return machine-readable JSON, never prose."
        ),
        llm=llm_quiz,
        tools=[],  # no tools: the required JSON format is embedded in the task prompt
        allow_delegation=False,
        **common,
    )

    evaluator = Agent(
        role="Leo Evaluator",
        goal=(
            "Grade the student's answers accurately, give constructive feedback, and decide "
            "whether the student is ready to move on or needs re-teaching."
        ),
        backstory=(
            "You are a supportive examiner. You grade objectively, use tools for arithmetic "
            "instead of guessing, and always begin feedback with what the student did well. "
            "You diagnose misconceptions from wrong answers and, when performance is below "
            "the pass threshold, give the Explainer precise guidance on what to re-teach."
        ),
        llm=llm_evaluator,
        tools=[grade_objective_answers],
        allow_delegation=False,
        **common,
    )

    return LeoAgents(coordinator, explainer, quiz_master, evaluator)