# quizcomp_ui/ui_components.py
from typing import List, Tuple, Dict, Any, Optional
import re
import streamlit as st


# LaTeX-aware rendering helpers


# 1) Environments: \begin{...} ... \end{...}
LATEX_ENV_RE = re.compile(r"(\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\})", re.DOTALL)

# 2) Display math: $$ ... $$ or \[ ... \]
DISPLAY_MATH_RE = re.compile(r"(\$\$.*?\$\$|\\\[.*?\\\])", re.DOTALL)

# 3) Inline math: $...$ or \( ... \)
INLINE_PAREN_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

# Lines/fragments we never want to display alone (common CSV artifacts)
# (expanded to include commas/semicolons/colons)
JUNK_LINE_RE = re.compile(r"^\s*(\${1,2}|\.|,|;|:|•|·)\s*$")

# LaTeX commands that often appear "bare" in CSV and should be inline math
BARE_LATEX_CMD_RE = re.compile(
    r"(\\left|\\right|\\cdot|\\times|\\frac|\\sqrt|\\sum|\\int|\\lim|\\log|\\ln|\\sin|\\cos|\\tan|\\leq|\\geq|\\neq|\\approx|\\infty|\\alpha|\\beta|\\gamma|\\theta|\\pi)"
)

# Wrap single bare LaTeX command chunks that are not already inside $...$ or \( ... \) or \[...\]
DOLLAR_BLOCK_RE = re.compile(r"\$.*?\$", re.DOTALL)
PAREN_BLOCK_RE  = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
BRACK_BLOCK_RE  = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)

def _wrap_bare_latex_inline(text: str) -> str:
    """
    If a text chunk contains LaTeX commands like \\left, \\cdot, etc.
    but is not already in math delimiters, wrap the whole chunk in $...$.
    This fixes cases like '\\left(1,2\\right)' showing with backslashes.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # If it's already clearly math-delimited, leave it.
    if "$" in t or "\\(" in t or "\\[" in t:
        return t

    # If it contains a LaTeX command, wrap it.
    if BARE_LATEX_CMD_RE.search(t):
        return f"${t}$"

    return t

def _strip_display_wrappers(block: str) -> str:
    b = block.strip()
    if b.startswith("$$") and b.endswith("$$"):
        return b[2:-2].strip()
    if b.startswith("\\[") and b.endswith("\\]"):
        return b[2:-2].strip()
    return b

def _drop_junk_lines(text: str) -> str:
    """
    Remove lines that are just '$', '$$', '.', ',', ';', ':', bullets, etc.
    These appear frequently in CSV exports and become visible after splitting.
    """
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").split("\n")
    kept: list[str] = []
    for ln in lines:
        if JUNK_LINE_RE.match(ln or ""):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()

def _escape_dangling_dollars(text: str) -> str:
    """
    If the text contains unmatched single '$' (not part of $...$),
    escape them so Markdown doesn't misinterpret.
    """
    if not text:
        return ""

    t = _drop_junk_lines(text)

    # If odd number of '$', escape all '$' to avoid breaking markdown
    # (we keep real $...$ blocks intact when well-formed; this is a safety net)
    if t.count("$") % 2 == 1:
        t = t.replace("$", r"\$")
    return t

def _normalize_text_for_markdown(text: str) -> str:
    """
    Make the *non-math* text readable without breaking MathJax.
    - Convert single newlines to spaces
    - Keep paragraph breaks
    - Drop junk lines/fragments
    """
    t = (text or "").replace("\r\n", "\n").strip()
    if not t:
        return ""

    # Remove garbage lines early
    t = _drop_junk_lines(t)
    if not t:
        return ""

    # Convert \( ... \) to $ ... $ (more consistent in Streamlit markdown)
    t = INLINE_PAREN_RE.sub(lambda m: f"${m.group(1).strip()}$", t)

    # Keep paragraph breaks, collapse single newlines
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Collapse extra spaces
    t = re.sub(r"[ \t]+", " ", t).strip()

    # Remove punctuation-only leftovers (after collapsing)
    if JUNK_LINE_RE.match(t):
        return ""

    # Escape dangling '$' to avoid broken markdown
    t = _escape_dangling_dollars(t)

    # Also remove cases where punctuation is isolated by spaces (",", ".")
    t = re.sub(r"\s+([,.;:])\s+", r"\1 ", t)   # " , " -> ", "
    t = re.sub(r"\s+([,.;:])$", r"\1", t)      # trailing " ,"
    t = re.sub(r"^([,.;:])\s+", r"", t)        # leading "," alone

    return t.strip()

def _render_display_math_left(expr: str):
    """
    Display math (normally centered) rendered using st.latex (centered by MathJax).
    We'll keep it as st.latex (best fidelity), BUT in practice:
    - the centering problem you reported is from INLINE math being sent to st.latex
    - display math is fine to keep centered (tables, big equations)
    """
    expr = (expr or "").strip()
    if not expr or JUNK_LINE_RE.match(expr):
        return
    st.latex(expr)

def _render_mixed_text_and_math(s: str):
    """
    Render a string that may contain:
    - normal text
    - inline math: $...$ or \( ... \)  (kept inline in markdown)
    - display math: $$...$$ or \[...\] (st.latex)
    - LaTeX environments (st.latex)
    """
    s = (s or "").replace("\r\n", "\n").strip()
    if not s:
        st.markdown("—")
        return

    # Pre-clean: remove junk lines so they never appear as standalone blocks
    s = _drop_junk_lines(s)
    if not s:
        st.markdown("—")
        return

    # Split out environments first (must be separate calls)
    chunks = LATEX_ENV_RE.split(s)

    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue

        chunk = chunk.strip()
        if JUNK_LINE_RE.match(chunk):
            continue

        # Environment => st.latex (display)
        if chunk.startswith("\\begin{") and "\\end{" in chunk:
            st.latex(chunk)
            continue

        # Split out display math blocks
        parts = DISPLAY_MATH_RE.split(chunk)

        # We try to keep text chunks as FEW markdown calls as possible to avoid line breaks.
        text_buffer: list[str] = []

        for part in parts:
            if not part or not part.strip():
                continue

            part = part.strip()
            if JUNK_LINE_RE.match(part):
                continue

            # Display math => flush text buffer, then latex
            if (part.startswith("$$") and part.endswith("$$")) or (part.startswith("\\[") and part.endswith("\\]")):
                # Flush pending text first (keeps punctuation attached, avoids lonely commas)
                if text_buffer:
                    merged = "\n\n".join([t for t in text_buffer if t.strip()])
                    merged = _normalize_text_for_markdown(merged)
                    if merged:
                        st.markdown(merged)
                    text_buffer = []

                body = _strip_display_wrappers(part).strip()
                if body and not JUNK_LINE_RE.match(body):
                    _render_display_math_left(body)
            else:
                # Normal text: if it contains bare LaTeX commands, wrap inline
                part2 = _wrap_bare_latex_inline(part)
                txt = _normalize_text_for_markdown(part2)
                if txt:
                    text_buffer.append(txt)

        # Flush remaining text at end of chunk
        if text_buffer:
            merged = "\n\n".join([t for t in text_buffer if t.strip()])
            merged = _normalize_text_for_markdown(merged)
            if merged:
                st.markdown(merged)

def render_text_block(s: str):
    """
    Public function used by the UI: best visual rendering of text + LaTeX.
    """
    s = (s or "").strip()
    if not s:
        st.markdown("—")
        return

    # Use the LaTeX-aware renderer (same behavior everywhere)
    _render_mixed_text_and_math(s)



def pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    """Return first non-empty field among keys."""
    for k in keys:
        v = d.get(k, None)
        if v is None:
            continue
        v = str(v).strip()
        if v and v.lower() not in ("nan", "none", "null"):
            return v
    return None

def _norm_ans(x: Optional[str]) -> str:
    return (x or "").strip()

def infer_correct_letter(mcq: Dict[str, Any]) -> Optional[str]:
    """
    Returns 'A'/'B'/'C'/'D' if we can infer it, else None.
    Handles cases where correct_option is 'A' or the actual option text.
    """
    corr = _norm_ans(pick_first(mcq, ["correct_option", "correct", "answer", "correctAnswer"]))
    if not corr:
        return None

    if corr.upper() in ("A", "B", "C", "D"):
        return corr.upper()

    A = _norm_ans(pick_first(mcq, ["option_a"]))
    B = _norm_ans(pick_first(mcq, ["option_b"]))
    C = _norm_ans(pick_first(mcq, ["option_c"]))
    D = _norm_ans(pick_first(mcq, ["option_d"]))

    if corr == A: return "A"
    if corr == B: return "B"
    if corr == C: return "C"
    if corr == D: return "D"
    return None

def normalize(weights: List[float]) -> List[float]:
    s = float(sum(weights))
    if s <= 0:
        return weights
    return [float(w) / s for w in weights]

def topic_selector(topics: List[Tuple[int, str, int]]):
    """
    Returns:
      vec: List[float] topic target distribution (same as before)
      chosen: List[str] selected topic names (used for generation listTopics)
    """
    st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Topic targets</div>", unsafe_allow_html=True)
    st.caption(
        "Select the topics you want in the quiz. Then assign a percentage target. "
        "Example: 50% means about half of the questions will be from this topic."
    )

    labels = [name for (_tid, name, _count) in topics]
    chosen = st.multiselect("Choose topics", labels, default=[])

    weights = []
    for name in chosen:
        weights.append(
            float(
                st.slider(
                    f"{name} (%)",
                    0, 100, 0, 5,
                    key=f"tw_{name}",
                    help="Percentage of questions in your quiz that should be from this topic."
                )
            )
        )

    vec = [0.0] * len(topics)
    if chosen and sum(weights) > 0:
        wnorm = normalize(weights)
        idx = {name: i for i, (_tid, name, _count) in enumerate(topics)}
        for name, w in zip(chosen, wnorm):
            vec[idx[name]] = float(w)

    return vec, chosen

def render_coverage(best_quiz: dict, topics, total_questions: int):
    st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Coverage Summary</div>", unsafe_allow_html=True)

    topic_lines = []
    for i, (_tid, name, _count) in enumerate(topics):
        v = float(best_quiz.get(f"topic_coverage_{i}", 0.0))
        if v > 0:
            topic_lines.append((name, v))

    diff_lines = []
    for i in range(6):
        v = float(best_quiz.get(f"difficulty_coverage_{i}", 0.0))
        if v > 0:
            diff_lines.append((f"Level {i+1}", v))

    if total_questions <= 0:
        total_questions = 0

    if topic_lines:
        st.write(f"Your quiz contains {total_questions} questions.")
        st.write("Topic distribution:")
        for name, v in topic_lines:
            pct = int(round(v * 100))
            approx_n = int(round(v * total_questions))
            st.write(f"- {pct}% of the questions are from {name} (about {approx_n} questions).")
    else:
        st.write("No topic coverage information available.")

    if diff_lines:
        st.write("Difficulty distribution:")
        for name, v in diff_lines:
            pct = int(round(v * 100))
            approx_n = int(round(v * total_questions))
            st.write(f"- {pct}% of the questions are {name} (about {approx_n} questions).")

def render_mcq_list(items: List[dict]):
    st.divider()
    st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Quiz Questions</div>", unsafe_allow_html=True)

    st.info(
        " **Note:** The questions are built in a way where the correct answer is always the first one, That's why it always shows as option A."
    )

    def _opt_row(letter: str, text: Optional[str]):
        col_l, col_r = st.columns([1, 25])
        with col_l:
            st.markdown(f"**{letter})**")
        with col_r:
            render_text_block(text or "—")

    for i, mcq in enumerate(items, start=1):
        topic_name = pick_first(mcq, ["topic_name", "topicName", "topic"]) or "Unknown topic"
        level = pick_first(mcq, ["difficulty", "difficulty_level", "Level", "level"]) or "?"
        qtext = pick_first(mcq, ["question", "Question", "statement", "prompt", "text"]) or ""

        A = pick_first(mcq, ["option_a", "answer_a", "A", "a"])
        B = pick_first(mcq, ["option_b", "answer_b", "B", "b"])
        C = pick_first(mcq, ["option_c", "answer_c", "C", "c"])
        D = pick_first(mcq, ["option_d", "answer_d", "D", "d"])

        st.markdown(
            f"<div style='font-size:1.05rem; font-weight:600; margin-top:1rem;'>"
            f"Q{i} — {topic_name} — Level {level}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if qtext:
            render_text_block(qtext)
        else:
            st.warning("Question text missing.")

        _opt_row("A", A)
        _opt_row("B", B)
        _opt_row("C", C)
        _opt_row("D", D)

        correct_letter = infer_correct_letter(mcq)
        correct_raw = pick_first(mcq, ["correct_option", "correct", "answer", "correctAnswer"])

        if correct_letter:
            st.markdown(f"**Correct answer:** {correct_letter}")
        elif correct_raw:
            st.markdown("**Correct answer:**")
            render_text_block(correct_raw)

        st.divider()

def difficulty_selector(num_difficulties: int = 6) -> List[float]:
    st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Difficulty levels</div>", unsafe_allow_html=True)
    st.caption(
        "Select difficulty levels and assign percentage targets. "
        "Example: 50% means about half of the questions will be of this difficulty.\n\n"
        "Difficulty meaning: 1 very easy, 2 easy, 3-4 medium, 5 hard, 6 very hard."
    )

    labels = [f"Level {i+1}" for i in range(num_difficulties)]
    chosen = st.multiselect("Choose difficulty levels", labels, default=[])

    weights = []
    for lab in chosen:
        weights.append(float(st.slider(f"{lab} (%)", 0, 100, 0, 5, key=f"dw_{lab}")))

    vec = [0.0] * num_difficulties
    if chosen and sum(weights) > 0:
        wnorm = normalize(weights)
        idx = {lab: i for i, lab in enumerate(labels)}
        for lab, w in zip(chosen, wnorm):
            vec[idx[lab]] = float(w)

    return vec

def render_best_quiz(best_quiz: dict, topics):
    st.markdown("<div style='font-size:1.2rem; font-weight:600;'>Best quiz</div>", unsafe_allow_html=True)

    quiz_id = best_quiz.get("quiz_id")
    target_match = best_quiz.get("targetMatch")

    st.write(f"Quiz id: {quiz_id}")
    if target_match is not None:
        st.write(f"Target match: {target_match}")

    mcq_cols = sorted(
        [k for k in best_quiz.keys() if k.startswith("mcq_")],
        key=lambda x: int(x.split("_")[1]),
    )
    mcq_ids = [best_quiz[c] for c in mcq_cols]
    st.write("MCQ ids in quiz:")
    st.code(", ".join(map(str, mcq_ids)))

    topic_named = {}
    for i, (_, name, _) in enumerate(topics):
        v = float(best_quiz.get(f"topic_coverage_{i}", 0.0))
        if v > 0:
            topic_named[name] = round(v, 4)

    diff_named = {}
    for i in range(6):
        v = float(best_quiz.get(f"difficulty_coverage_{i}", 0.0))
        if v > 0:
            diff_named[f"Level {i+1}"] = round(v, 4)

    if topic_named:
        st.write("Topic coverage:")
        st.json(topic_named)

    if diff_named:
        st.write("Difficulty coverage:")
        st.json(diff_named)
