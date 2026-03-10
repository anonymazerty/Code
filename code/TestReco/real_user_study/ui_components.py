import time
import datetime as dt
import streamlit as st


# -------------------------
# Global header
# -------------------------

def header():
        # Add CSS to discourage copying/selection and a centered header.
        # Note: this is a best-effort client-side measure — determined users can still copy via devtools,
        # view-source, or other tools.
        st.markdown(
            """
            <style>
            /* Disable text selection across the app */
            body, .stApp, .block-container, .main, .element-container, .css-1d391kg {
                -webkit-user-select: none !important;
                -moz-user-select: none !important;
                -ms-user-select: none !important;
                user-select: none !important;
            }
            /* Prevent selection highlighting */
            ::selection { background: transparent; }
            /* Minimal right-click hint (JS may be required to fully block context menu) */
            </style>

            <div style="text-align:center; line-height:1; margin-top:-24px; margin-bottom:24px; padding:0;">
                <h1 style="margin:0; padding:0;">Adaptive Test Recommendation</h1>
                <h1 style="margin:0; padding:0;">System Study</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------
# Landing / Overview (before login)
# -------------------------

def landing_overview():
    """
    Landing page text shown BEFORE login/registration.
    Returns True if the user clicks "Next".
    """

    st.markdown("""
Welcome! This research study evaluates whether an **adaptive test recommendation system** selects questions that match a learner’s level over time.

**Topic:** Fundamental Mathematics (general math knowledge, primarily **algebra** and **geometry**).  
You do **not** need to be a math expert, but you should be comfortable with standard school-level math reasoning.

### Study structure:
1. **Pre-qualification test (12 questions):** a short screening test across multiple difficulty levels.  
2. **Learning phase (adaptive recommendations):** the system recommends questions based on your answers.  
3. **Final questionnaire:** brief qualitative feedback about your experience.

Click **Next** to continue.
""")

    st.markdown(
        """
        <p style="color:red; font-weight:900;"> Note:</p> You may stop and abandon the study at any time by clicking the <strong>End session now</strong> button in the study interface. Please note that if you stop the session before completing the final questionnaire, you will not receive the same payment as participants who finish the full session.
        """,
        unsafe_allow_html=True,
    )

    return st.button("Next")


# -------------------------
# Consent block
# -------------------------

def consent_block():
        st.markdown(
                """
                <h3>Consent and data use</h3>

                <p>This study is conducted for <strong>research purposes</strong> to evaluate an adaptive test recommendation system.</p>

                <p><strong>During the session, we will record:</strong></p>
                <ul>
                    <li>your answers (correct/incorrect)</li>
                    <li>the questions shown and the system’s recommendation decisions</li>
                    <li><strong>response times</strong> (time taken to answer each question)</li>
                    <li>interaction events needed to run the study (e.g., phase transitions)</li>
                </ul>

                <div style="margin-top:8px;">
                    <p style="color:#2b6cb0; font-weight:700; margin:0;">Privacy</p>
                    <ul>
                        <li>No personally identifying data is required beyond a <strong>username</strong>.</li>
                        <li>Login credentials are not retained for identifying participants and will not be used to link you personally to published study results.</li>
                    </ul>
                </div>

                <div style="margin-top:6px;">
                    <p style="color:#2b6cb0; font-weight:700; margin:0;">Voluntary participation</p>
                    <ul>
                        <li>You may stop at any time by clicking the <strong>End session now</strong> button.</li>
                        <li>If you do not complete the full session, the payment will be reduced.</li>
                    </ul>
                </div>

                <div style="margin-top:6px;">
                    <p style="color:#2b6cb0; font-weight:700; margin:0;">Sign-up &amp; login</p>
                    <ul>
                        <li>Sign-up/login is only used to register participants for a session.</li>
                        <li>No special password requirements or restrictions.</li>
                        <li>You do not need to provide your full legal name.</li>
                    </ul>
                </div>

                <div style="margin-top:6px;">
                    <p style="color:#2b6cb0; font-weight:700; margin:0;">Compensation</p>
                    <p>You will be paid according to the <strong>time you spend</strong> on the study and the <strong>number of questions</strong> you answer. To receive the <strong>full amount</strong>, you must reach and complete the final questionnaire phase.</p>
                </div>

                <p>Please confirm below if you agree to participate.</p>
                """,
                unsafe_allow_html=True,
        )
        return st.checkbox("I consent to participate.", value=False)


# -------------------------
# Phase intro helpers
# -------------------------

def qualification_intro(n_questions: int = 12):
    """
    Text shown BEFORE the pre-qualification test starts.
    Returns True if the user clicks "Start qualification test".
    """
    st.markdown(f"""
## Phase 1: Pre-qualification test ({n_questions} questions)

You will answer **{n_questions} predefined questions** covering **Fundamental Mathematics**, mainly:
- **Algebra** fundamentals
- **Geometry** fundamentals

The questions span **different difficulty levels**. This phase helps us ensure participants have the minimum background needed to meaningfully interact with the system.

### Important notes
- Please avoid random guessing. **Patterns consistent with random guessing will be detected**.
- Take your time: your **response time is recorded** for analysis.

When you are ready, click **Start qualification test**. Good luck!
""")
    return st.button("Start qualification test")


def qualification_results_block(
    passed: bool,
    score: int,
    total: int,
    mastery: float | None = None,
    guessing_info: dict | None = None,
):
    """
    UI block for showing qualification results.
    Returns:
      - True if user clicks "Continue to learning phase" (only if passed)
      - False otherwise
    """
    st.markdown("## Qualification test results")

    # Show score as ratio only (e.g., 0.75), not "9/12"
    ratio = 0.0
    try:
        ratio = float(score) / float(total) if total else 0.0
    except Exception:
        ratio = 0.0

    if passed:
        st.success("You passed the pre-qualification phase and can continue to the learning phase.")
    # Only show "Guessing detected" if provided
    if guessing_info is not None:
        st.markdown("### Results:")
        is_guessing = bool(guessing_info.get("is_guessing", False))
        st.write(f"- **Your prequalification Score:** {ratio:.2f}")
        st.write(f"- **Guessing detected:** {is_guessing}")

    if passed:
        return st.button("Continue to learning phase")

    st.error("You did not meet the minimum threshold for this session. Thank you for your time.")
    st.markdown("You may close this page now.")
    return False




def learning_intro(steps_n: int = 10, max_q_per_step: int = 5):
    """
    Text shown BEFORE the learning phase starts.
    Returns True if the user clicks "Start learning process".
    """
    st.markdown(f"""
## Phase 2: Learning phase

In this phase, you will go through **{steps_n} steps**.  
At each step, you may receive up to **{max_q_per_step} questions** (sometimes fewer).

The system will recommend questions based on your previous answers.  
Your task is simply to do your best and answer as accurately as you can.

### Important notes
- Please avoid random answering: the goal is **not to judge you**, but to evaluate whether the **recommendation system** adapts to your level appropriately.
- Your **answer correctness** and **time taken** are recorded to analyze the system’s behavior.

Click **Start learning process** when you’re ready.
""")
    return st.button("Start learning process")


def learning_completion_block(mastery_value: float | None = None, mastery_series=None):
    """
    Text shown AFTER finishing the learning phase and BEFORE the survey.
    Optionally plots mastery evolution if a series is provided.
    Returns True if the user clicks "Continue to final questionnaire".
    """
    st.success("Congratulations — you have completed the learning phase.")

    if mastery_value is not None:
        st.markdown(f"**Your estimated mastery level:** `{float(mastery_value):.2f}`")

    if mastery_series is not None:
        try:
            # Streamlit can plot lists, dicts, pandas series, etc.
            st.line_chart(mastery_series)
        except Exception:
            pass

    st.markdown("""
You are now invited to answer a short **final questionnaire** about your experience with the recommendation system.
""")
    return st.button("Continue to final questionnaire")


def survey_intro():
    """
    Text shown BEFORE the survey questions.
    Returns True if the user clicks "Start questionnaire".
    """
    st.markdown("""
## Phase 3: Your feedback

Your feedback is important for our analysis.  
Please answer carefully and avoid answering randomly.

This questionnaire focuses on:
- perceived difficulty and appropriateness of recommended questions,
- overall learning experience and satisfaction.

Please answer each question on a **1–5 scale**, where:

- **1** = negative assessment (e.g., *very difficult*, *very demanding*, *not satisfied*, *low trust*)
- **5** = positive assessment (e.g., *very easy*, *not demanding*, *very satisfied*, *high trust*)


Click **Start questionnaire** to continue.
""")
    return st.button("Start questionnaire")


def study_complete_block():
    """
    Final message displayed at the end of the study (after survey submission).
    """
    st.markdown("""
## Thank you!

Thank you for participating in this research study. Your responses will help us evaluate and improve adaptive recommendation methods for math practice.

You may now close this page.
""")



def inject_likert_css():
    st.markdown(
        """
<style>
/* Make Streamlit radios look like 5 rectangular buttons */
div[data-testid="stRadio"] > div[role="radiogroup"]{
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
}

/* Each option container */
div[data-testid="stRadio"] label[data-baseweb="radio"]{
    flex: 1 1 0;
    margin: 0 !important;
}

/* Hide the default radio circle */
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child{
    display: none !important;
}

/* Rectangle styling for the clickable area */
div[data-testid="stRadio"] label[data-baseweb="radio"] input + div{
    width: 100%;
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 10px;
    padding: 0.55rem 0;
    justify-content: center;
    text-align: center;
    cursor: pointer;
}

/* Selected = RED */
div[data-testid="stRadio"] label[data-baseweb="radio"] input:checked + div{
    background: #ff4b4b !important;
    color: white !important;
    border-color: #ff4b4b !important;
    font-weight: 700;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def likert(prompt: str, key: str):
    """
    Likert 1..5 with:
      - NO default selection
      - NO visible widget label
      - selected option turns RED (via injected CSS)
    Returns:
      int in [1..5] or None
    """
    st.markdown(prompt)

    options = ["1", "2", "3", "4", "5"]

    # Try modern Streamlit: index=None => no preselection
    try:
        val = st.radio(
            "Likert",
            options=options,
            index=None if st.session_state.get(key) is None else (int(st.session_state[key]) - 1),
            key=key,
            horizontal=True,
            label_visibility="collapsed",
        )
        return None if val is None else int(val)

    except TypeError:
        # Fallback for older Streamlit that doesn't support index=None/horizontal
        # Use button-based approach; selected option is marked with a red badge.
        if key not in st.session_state:
            st.session_state[key] = None

        cols = st.columns(5)
        for i in range(1, 6):
            selected = (st.session_state[key] == i)
            txt = f"**{i}**" + ("  🔴" if selected else "")
            if cols[i - 1].button(txt, key=f"{key}__btn_{i}", use_container_width=True):
                st.session_state[key] = i
                st.rerun()

        return st.session_state[key]


def render_question(q: dict, key_prefix: str):
    """
    q follows BenchmarkLoader output:
      - q["text"], q["options"], q["answer"], q["id"], q["topic"]
    """
    st.markdown(q["text"])
    options = q.get("options", [])
    choice = st.radio("Choose one:", options, key=f"{key_prefix}_choice")
    choice_idx = options.index(choice) if choice in options else -1
    correct_idx = int(q.get("answer", -1))
    is_correct = (choice_idx == correct_idx)
    return choice_idx, correct_idx, is_correct



def ms_since(t0: float) -> int:
    return int((time.time() - t0) * 1000)
