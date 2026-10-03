"""
Leo: Multi-Agent AI Tutor - Streamlit UI.

Run from the project root:   streamlit run app.py

How the UI maps onto the workflow (src/workflow.py)
---------------------------------------------------
* The ``LeoTutorWorkflow`` object lives in ``st.session_state`` so the lesson, quiz,
  attempts and handoff log survive Streamlit's re-runs (session state).
* Buttons never call the LLM directly. They queue an action (``st.session_state.pending``)
  and re-run; the next run executes it behind a spinner while a live "agent tracker"
  shows which agent is working, then re-runs again to draw the new phase.
* Human-in-the-loop gates are real phases of the workflow, rendered as forms with
  Approve / Revise buttons and free-text feedback.
* A low score sends the student back to gate 1 with a visible "feedback loop" banner.
"""

from __future__ import annotations

import os

# Set before CrewAI is imported: no telemetry / trace-sharing prompts inside Streamlit.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import hashlib
import json
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

from src.models import Difficulty
from src.workflow import AttemptRecord, LeoTutorWorkflow, Phase, WorkflowError, WorkflowEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENTS: List[Tuple[str, str, str]] = [
    ("Coordinator", "🧭", "Understands your request"),
    ("Explainer", "📘", "Teaches the topic"),
    ("Quiz Master", "📝", "Builds the quiz"),
    ("Evaluator", "🎯", "Grades & gives feedback"),
]
AGENT_NAMES = {name for name, _, _ in AGENTS}

PHASE_STEP = {
    Phase.INTAKE: (0.0, "Step 1 of 5 · Tell Leo what you want to learn"),
    Phase.NEEDS_CLARIFICATION: (0.0, "Step 1 of 5 · Leo needs a little more detail"),
    Phase.EXPLANATION_REVIEW: (0.25, "Step 2 of 5 · Review the explanation (human-in-the-loop gate 1)"),
    Phase.QUIZ_REVIEW: (0.5, "Step 3 of 5 · Review the quiz (human-in-the-loop gate 2)"),
    Phase.AWAITING_ANSWERS: (0.75, "Step 4 of 5 · Take the quiz"),
    Phase.COMPLETE: (1.0, "Step 5 of 5 · Results"),
}

SPINNER_TEXT = {
    "start": "Coordinator is checking your request, then the Explainer will teach it…",
    "revise_explanation": "Explainer is rewriting the explanation with your feedback…",
    "approve_explanation": "Quiz Master is writing your quiz…",
    "regenerate_quiz": "Quiz Master is writing a different quiz…",
    "approve_quiz": "Getting your quiz ready…",
    "submit_answers": "Evaluator is grading your answers…",
}

KIND_ICON = {"start": "▶️", "step": "💭", "done": "✅", "handoff": "🔀", "warning": "⚠️", "error": "❌", "info": "ℹ️"}

CSS = """
<style>
.leo-row{display:flex;gap:.4rem;align-items:stretch;flex-wrap:wrap;margin:.25rem 0 .75rem 0}
.leo-arrow{align-self:center;opacity:.5;font-size:1.2rem}
.leo-card{flex:1 1 150px;border:2px solid rgba(128,128,128,.35);border-radius:12px;padding:.55rem .8rem}
.leo-card .t{font-weight:700}
.leo-card .s{font-size:.78rem;opacity:.7}
.leo-card .st{font-size:.85rem;margin-top:.25rem}
.leo-card .n{font-size:.75rem;opacity:.7;margin-top:.25rem;font-style:italic}
.leo-card.active{border-color:#f59e0b;background:rgba(245,158,11,.14);animation:leo-pulse 1.4s ease-in-out infinite}
.leo-card.done{border-color:#16a34a;background:rgba(22,163,74,.10)}
.leo-card.waiting{border-color:#3b82f6;background:rgba(59,130,246,.12)}
@keyframes leo-pulse{0%,100%{box-shadow:0 0 0 0 rgba(245,158,11,.45)}50%{box-shadow:0 0 0 8px rgba(245,158,11,0)}}
</style>
"""


# ---------------------------------------------------------------------------
# Rendering helpers: agent tracker + log
# ---------------------------------------------------------------------------
def _escape_md(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#$~|])", r"\\\1", text)


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pipeline_html(
    active: Optional[str],
    done: Set[str],
    waiting: Optional[str] = None,
    waiting_text: str = "⏸ waiting for you",
    last_step: Optional[Dict[str, str]] = None,
) -> str:
    """Four agent cards: working (pulsing amber), done (green), waiting for you (blue), idle."""
    last_step = last_step or {}
    cards = []
    for name, icon, role in AGENTS:
        if name == active:
            cls, status = "active", "⚙️ working…"
        elif name == waiting:
            cls, status = "waiting", waiting_text
        elif name in done:
            cls, status = "done", "✅ done"
        else:
            cls, status = "idle", "· idle"
        note = ""
        if cls == "active" and last_step.get(name):
            note = f'<div class="n">{_escape_html(" ".join(last_step[name].split())[:90])}</div>'
        cards.append(
            f'<div class="leo-card {cls}"><div class="t">{icon} {name}</div>'
            f'<div class="s">{role}</div><div class="st">{status}</div>{note}</div>'
        )
    return '<div class="leo-row">' + '<div class="leo-arrow">→</div>'.join(cards) + "</div>"


def static_view(phase: Phase) -> Tuple[Set[str], Optional[str], str]:
    """Tracker state when nothing is running: (done agents, agent waiting on the student, label)."""
    table = {
        Phase.INTAKE: (set(), None, ""),
        Phase.NEEDS_CLARIFICATION: (set(), "Coordinator", "⏸ awaiting your reply"),
        Phase.EXPLANATION_REVIEW: ({"Coordinator"}, "Explainer", "⏸ awaiting your review"),
        Phase.QUIZ_REVIEW: ({"Coordinator", "Explainer"}, "Quiz Master", "⏸ awaiting your review"),
        Phase.AWAITING_ANSWERS: ({"Coordinator", "Explainer", "Quiz Master"}, "Evaluator", "⏸ awaiting your answers"),
        Phase.COMPLETE: (set(AGENT_NAMES), None, ""),
    }
    return table[phase]


def log_markdown(events: List[WorkflowEvent], limit: int = 40) -> str:
    if not events:
        return "_No activity yet._"
    lines = []
    for e in events[-limit:]:
        stamp = e.timestamp.astimezone().strftime("%H:%M:%S")
        msg = _escape_md(" ".join(e.message.split())[:150])
        lines.append(f"`{stamp}` {KIND_ICON.get(e.kind, '•')} **{e.agent}** — {msg}")
    return "\n\n".join(lines)


class LiveUI:
    """
    Receives workflow events and repaints the tracker/log placeholders while an action runs.
    Events may arrive from CrewAI callbacks on another thread; those are only recorded, and
    the screen is repainted from the Streamlit script thread (calling st.* elsewhere fails).
    """

    def __init__(self) -> None:
        self.events: List[WorkflowEvent] = []
        self.active: Optional[str] = None
        self.done: Set[str] = set()
        self.last_step: Dict[str, str] = {}
        self._pipeline_ph: Any = None
        self._log_ph: Any = None
        self._thread_id: Optional[int] = None

    def begin(self, pipeline_ph: Any, log_ph: Any, base_done: Set[str]) -> None:
        self._pipeline_ph, self._log_ph = pipeline_ph, log_ph
        self._thread_id = threading.get_ident()
        self.active, self.done, self.last_step = None, set(base_done), {}
        self._repaint()

    def end(self) -> None:
        self.active = None
        self._pipeline_ph = self._log_ph = self._thread_id = None

    def handle(self, event: WorkflowEvent) -> None:
        self.events.append(event)
        if event.kind == "start":
            self.active = event.agent
        elif event.kind == "step":
            self.last_step[event.agent] = event.message
        elif event.kind in ("done", "error"):
            if event.kind == "done":
                self.done.add(event.agent)
            if self.active == event.agent:
                self.active = None
        self._repaint()

    def _repaint(self) -> None:
        if self._thread_id != threading.get_ident():
            return
        try:
            if self._pipeline_ph is not None:
                self._pipeline_ph.markdown(
                    pipeline_html(self.active, self.done, None, "", self.last_step), unsafe_allow_html=True
                )
            if self._log_ph is not None:
                self._log_ph.markdown(log_markdown(self.events))
        except Exception:  # never let a repaint problem break the tutoring flow
            pass


# ---------------------------------------------------------------------------
# Session state / workflow lifecycle
# ---------------------------------------------------------------------------
def get_workflow() -> Tuple[LeoTutorWorkflow, LiveUI]:
    if "wf" not in st.session_state:
        live = LiveUI()
        try:
            st.session_state.wf = LeoTutorWorkflow(on_event=live.handle)
        except Exception as exc:  # missing API key, missing provider package, ...
            st.error(f"Leo could not start: {exc}")
            st.info("Check your `.env` file (see `.env.example`): `LEO_MODEL` and the matching API key.")
            st.stop()
        st.session_state.live = live
        st.session_state.explanations = []  # [{"label": str, "text": str}]
    return st.session_state.wf, st.session_state.live


def queue(action: str, **kwargs: Any) -> None:
    """Queue a workflow action; it runs on the next script run behind a spinner."""
    st.session_state.pending = (action, kwargs)
    st.session_state.pop("flash_error", None)
    st.rerun()


def sync_explanations(wf: LeoTutorWorkflow, action: str) -> None:
    """Keep every explanation version for the history panel (the workflow only keeps the latest)."""
    text = wf.state.explanation
    history: List[Dict[str, str]] = st.session_state.explanations
    if not text or (history and history[-1]["text"] == text):
        return
    if action == "submit_answers":
        label = f"Re-teach #{wf.state.reteach_count} (feedback loop)"
    elif action == "revise_explanation":
        label = f"Revision {sum('Revision' in h['label'] for h in history) + 1} (your feedback)"
    else:
        label = "Initial explanation"
    history.append({"label": label, "text": text})


def run_pending(wf: LeoTutorWorkflow, live: LiveUI, pipeline_ph: Any, log_ph: Any, base_done: Set[str]) -> None:
    action, kwargs = st.session_state.pop("pending")
    st.session_state.last_action = (action, kwargs)
    live.begin(pipeline_ph, log_ph, base_done)
    error: Optional[str] = None
    try:
        with st.spinner(SPINNER_TEXT.get(action, "Leo is working…")):
            getattr(wf, action)(**kwargs)
        sync_explanations(wf, action)
    except WorkflowError as exc:
        error = str(exc)
    except Exception as exc:  # unexpected: show it, keep the session alive
        error = f"Unexpected error: {exc}"
    finally:
        live.end()
    st.session_state.flash_error = error
    st.rerun()


def reset_session() -> None:
    wf, live = st.session_state.wf, st.session_state.live
    wf.reset()
    live.events.clear()
    st.session_state.explanations = []
    for key in ("pending", "flash_error", "last_action", "last_request"):
        st.session_state.pop(key, None)
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(wf: LeoTutorWorkflow, live: LiveUI, busy: bool) -> Any:
    with st.sidebar:
        st.header("⚙️ Session")
        s = wf.settings
        st.caption(f"Model: `{s.model}`")
        st.caption(f"Pass mark: **{s.pass_threshold:g}%** · max re-teach rounds: **{s.max_reteach_loops}**")
        st.caption(f"Re-teach rounds used: **{wf.state.reteach_count}/{s.max_reteach_loops}**")

        st.subheader("📡 Live agent log")
        box = st.container(height=300)
        with box:
            log_ph = st.empty()
            log_ph.markdown(log_markdown(live.events))

        st.subheader("🔀 Handoff timeline")
        if not wf.state.handoffs:
            st.caption("Handoffs between agents and you will appear here.")
        for h in wf.state.handoffs[-14:]:
            stamp = h.timestamp.astimezone().strftime("%H:%M:%S")
            st.markdown(f"`{stamp}` **{h.from_agent}** → **{h.to_agent}**")
            st.caption(h.reason + (f"  ·  _{', '.join(h.payload_keys)}_" if h.payload_keys else ""))

        st.divider()
        if st.button("🧹 Reset session", disabled=busy, use_container_width=True):
            reset_session()
    return log_ph


# ---------------------------------------------------------------------------
# Phase renderers
# ---------------------------------------------------------------------------
def render_intake(wf: LeoTutorWorkflow) -> None:
    if wf.phase == Phase.NEEDS_CLARIFICATION and wf.clarification:
        st.info(f"🧭 **Coordinator:** {wf.clarification}")
    with st.form("intake_form"):
        text = st.text_area(
            "What would you like to learn?",
            value=st.session_state.get("last_request", ""),
            placeholder="e.g. Explain photosynthesis for a beginner",
            height=100,
        )
        level_label = st.selectbox("Your level", ["Let Leo decide", "Beginner", "Intermediate", "Advanced"])
        submitted = st.form_submit_button("🚀 Start learning", type="primary")
    if submitted:
        if not text.strip():
            st.warning("Please type a topic first.")
            return
        st.session_state.last_request = text.strip()
        level = None if level_label.startswith("Let") else Difficulty(level_label.lower())
        queue("start", student_input=text, learner_level=level)


def render_attempt_feedback(attempt: AttemptRecord) -> None:
    fb_by_id = {f.question_id: f for f in attempt.evaluation.per_question}
    for q in attempt.quiz.questions:
        fb = fb_by_id.get(q.id)
        score = fb.score if fb else 0.0
        icon = "✅" if score >= 1 else ("🟡" if score > 0 else "❌")
        with st.container(border=True):
            st.markdown(f"{icon} **{q.id.upper()}** · {q.question}")
            st.markdown(f"**Your answer:** {attempt.answers.get(q.id) or '_(blank)_'}")
            if score < 1:
                st.markdown(f"**Correct answer:** {q.correct_answer}")
            if fb:
                st.markdown(f"💬 {fb.feedback}")
                if fb.misconception:
                    st.caption(f"Possible misconception: {fb.misconception}")
            st.caption(f"Why: {q.explanation}")


def render_gate1(wf: LeoTutorWorkflow) -> None:
    """Human-in-the-loop gate 1: review the explanation before any quiz is created."""
    state = wf.state
    if state.reteach_count > 0 and state.evaluation is not None:
        ev = state.evaluation
        weak = ", ".join(ev.weak_concepts) if ev.weak_concepts else "the topics you missed"
        st.warning(
            f"🔁 **Feedback loop · re-teach {state.reteach_count} of {wf.settings.max_reteach_loops}.** "
            f"You scored **{ev.overall_score_pct:.0f}%** (pass mark {wf.settings.pass_threshold:g}%), so the "
            f"Evaluator sent you back to the Explainer. Focus: **{weak}**."
        )
        if ev.reteach_focus:
            with st.expander("What the Evaluator told the Explainer"):
                st.write(ev.reteach_focus)
        if wf.attempts:
            with st.expander(f"📊 Your last attempt: {wf.attempts[-1].evaluation.overall_score_pct:.0f}% — see feedback"):
                render_attempt_feedback(wf.attempts[-1])

    st.subheader(f"📘 {state.topic.title()}")
    st.caption(f"Level: {state.learner_level.value}")
    st.markdown(state.explanation or "_(no explanation yet)_")

    st.divider()
    st.markdown("#### 🧑‍🏫 Your turn — human-in-the-loop review")
    st.caption(
        "Happy with this? Approve it and Leo builds a quiz. Not quite right? Tell the Explainer what to change. "
        "You can revise as many times as you like."
    )
    with st.form("gate1_form"):
        c1, c2 = st.columns(2)
        revise_notes = c1.text_area(
            "Explain it differently…",
            placeholder="e.g. Simpler words, skip ATP/NADPH, add one more example",
            height=110,
        )
        quiz_notes = c2.text_area(
            "Focus the quiz on… (optional)",
            placeholder="e.g. Light reactions and stomata",
            height=110,
        )
        b1, b2 = st.columns(2)
        revise = b1.form_submit_button("✏️ Revise explanation", use_container_width=True)
        approve = b2.form_submit_button("✅ Approve & build quiz", type="primary", use_container_width=True)
    if revise:
        if not revise_notes.strip():
            st.warning("Tell Leo what to change first (left box).")
        else:
            queue("revise_explanation", notes=revise_notes)
    if approve:
        queue("approve_explanation", notes=quiz_notes.strip() or None)


def render_gate2(wf: LeoTutorWorkflow) -> None:
    """Human-in-the-loop gate 2: preview the quiz (answers hidden) before taking it."""
    quiz = wf.state.quiz
    if quiz is None:
        st.error("No quiz in the session. Please reset and try again.")
        return
    st.subheader("📝 Quiz preview")
    st.caption(f"{len(quiz.questions)} questions · {quiz.total_points} points · {quiz.difficulty.value}")
    for item in quiz.student_view():
        with st.container(border=True):
            kind = item["question_type"].replace("_", " ")
            st.markdown(f"**{item['id'].upper()}** · {item['question']}")
            for opt in item["options"]:
                st.markdown(f"- {opt}")
            st.caption(f"{kind} · concept: {item['concept_tag']} · {item['points']} pt")

    st.divider()
    st.markdown("#### 🧑‍🏫 Your turn — human-in-the-loop review")
    with st.form("gate2_form"):
        notes = st.text_area(
            "Want a different quiz? Tell the Quiz Master…",
            placeholder="e.g. More true/false, fewer multiple choice, focus on definitions",
            height=90,
        )
        b1, b2 = st.columns(2)
        regenerate = b1.form_submit_button("🔄 Generate a different quiz", use_container_width=True)
        approve = b2.form_submit_button("✅ Approve — I'm ready", type="primary", use_container_width=True)
    if regenerate:
        queue("regenerate_quiz", notes=notes.strip() or None)
    if approve:
        queue("approve_quiz")


def render_quiz_form(wf: LeoTutorWorkflow) -> None:
    quiz = wf.state.quiz
    if quiz is None:
        st.error("No quiz in the session. Please reset and try again.")
        return
    token = hashlib.md5(quiz.model_dump_json().encode()).hexdigest()[:8]  # fresh widgets for each new quiz
    st.subheader("✍️ Answer the quiz")
    if wf.state.reteach_count:
        st.caption(f"Attempt {len(wf.attempts) + 1} — after re-teaching round {wf.state.reteach_count}.")
    with st.form("quiz_form"):
        values: Dict[str, Optional[str]] = {}
        for q in quiz.questions:
            label = f"**{q.id.upper()}.** {q.question}"
            key = f"ans_{token}_{q.id}"
            if q.question_type.value == "short_answer":
                values[q.id] = st.text_area(label, key=key, height=90, placeholder="Write your answer in your own words")
            else:
                values[q.id] = st.radio(label, q.options, index=None, key=key, horizontal=q.question_type.value == "true_false")
            st.write("")
        submitted = st.form_submit_button("📤 Submit answers", type="primary")
    if submitted:
        answers = {qid: (val or "").strip() for qid, val in values.items()}
        if not any(answers.values()):
            st.warning("Answer at least one question before submitting.")
            return
        queue("submit_answers", answers=answers)


def build_report(wf: LeoTutorWorkflow) -> str:
    report = {
        "topic": wf.state.topic,
        "learner_level": wf.state.learner_level.value,
        "reteach_rounds_used": wf.state.reteach_count,
        "loop_exhausted": wf.loop_exhausted,
        "attempts": [
            {
                "attempt": a.attempt_number,
                "score_pct": a.evaluation.overall_score_pct,
                "recommendation": a.evaluation.recommendation,
                "weak_concepts": a.evaluation.weak_concepts,
                "answers": a.answers,
                "feedback": [f.model_dump() for f in a.evaluation.per_question],
                "summary": a.evaluation.summary,
            }
            for a in wf.attempts
        ],
        "handoffs": [h.model_dump(mode="json") for h in wf.state.handoffs],
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def render_complete(wf: LeoTutorWorkflow) -> None:
    ev = wf.state.evaluation
    if ev is None or not wf.attempts:
        st.error("No evaluation found. Please reset and try again.")
        return
    threshold = wf.settings.pass_threshold
    passed = ev.overall_score_pct >= threshold
    if passed:
        st.success("🎉 You passed! Great work.")
    else:
        st.warning(
            f"You finished below the pass mark. Leo used all {wf.settings.max_reteach_loops} re-teach rounds — "
            "review the weak concepts below and try a new session."
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Final score", f"{ev.overall_score_pct:.0f}%", delta=f"{ev.overall_score_pct - threshold:+.0f} vs pass mark")
    c2.metric("Quiz attempts", len(wf.attempts))
    c3.metric("Re-teach rounds", wf.state.reteach_count)

    st.markdown(f"**Evaluator's summary:** {ev.summary}")
    if ev.strengths:
        st.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in ev.strengths))
    if ev.weak_concepts:
        st.markdown("**Worth reviewing**\n" + "\n".join(f"- {w}" for w in ev.weak_concepts))

    if len(wf.attempts) > 1:
        st.markdown("**Score by attempt**")
        df = pd.DataFrame(
            {"Score %": [a.evaluation.overall_score_pct for a in wf.attempts]},
            index=[f"Attempt {a.attempt_number}" for a in wf.attempts],
        )
        st.bar_chart(df)

    st.subheader("Question-by-question feedback")
    render_attempt_feedback(wf.attempts[-1])

    b1, b2 = st.columns(2)
    b1.download_button(
        "⬇️ Download session report (JSON)",
        data=build_report(wf),
        file_name="leo_session_report.json",
        mime="application/json",
        use_container_width=True,
    )
    if b2.button("🔄 Learn something new", type="primary", use_container_width=True):
        reset_session()


def render_history(wf: LeoTutorWorkflow) -> None:
    explanations = st.session_state.explanations
    if not explanations and not wf.attempts:
        return
    with st.expander("📚 Learning history (this session)"):
        tab_expl, tab_att = st.tabs(["Explanations", "Quiz attempts"])
        with tab_expl:
            if not explanations:
                st.caption("Nothing yet.")
            for item in explanations:
                st.markdown(f"**{item['label']}**")
                st.markdown(item["text"])
                st.divider()
        with tab_att:
            if not wf.attempts:
                st.caption("No quiz attempts yet.")
            for a in wf.attempts:
                weak = f" · weak: {', '.join(a.evaluation.weak_concepts)}" if a.evaluation.weak_concepts else ""
                st.markdown(
                    f"**Attempt {a.attempt_number}:** {a.evaluation.overall_score_pct:.0f}% "
                    f"→ {a.evaluation.recommendation}{weak}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Leo · Multi-Agent AI Tutor", page_icon="🦁", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    wf, live = get_workflow()
    busy = "pending" in st.session_state
    log_ph = render_sidebar(wf, live, busy)

    st.title("🦁 Leo — Multi-Agent AI Tutor")
    st.caption(
        "Coordinator → Explainer → Quiz Master → Evaluator · human review at every gate · "
        "automatic re-teaching when your score is low"
    )

    pipeline_ph = st.empty()
    done, waiting, waiting_text = static_view(wf.phase)
    pipeline_ph.markdown(pipeline_html(None, done, waiting, waiting_text), unsafe_allow_html=True)

    value, label = PHASE_STEP[wf.phase]
    st.progress(value, text=label)

    if busy:  # run the queued action; the tracker above updates live, then the app re-runs
        run_pending(wf, live, pipeline_ph, log_ph, done)
        return

    error = st.session_state.get("flash_error")
    if error:
        st.error(f"❌ {error}")
        cols = st.columns([1, 1, 4])
        if st.session_state.get("last_action") and cols[0].button("🔁 Retry"):
            action, kwargs = st.session_state.last_action
            queue(action, **kwargs)
        if cols[1].button("Dismiss"):
            st.session_state.pop("flash_error", None)
            st.rerun()

    renderers = {
        Phase.INTAKE: render_intake,
        Phase.NEEDS_CLARIFICATION: render_intake,
        Phase.EXPLANATION_REVIEW: render_gate1,
        Phase.QUIZ_REVIEW: render_gate2,
        Phase.AWAITING_ANSWERS: render_quiz_form,
        Phase.COMPLETE: render_complete,
    }
    renderers[wf.phase](wf)
    render_history(wf)


main()
