# Survey questions prompt template
SURVEY_QUESTIONS_PROMPT = """
Please rate each statement on a scale of 1-5:

Q1 - Feeling of Accomplishment:
"How successful do you feel you were in building a quiz that matches your needs (topics and difficulty)?"

Q2 - Effort Required:
"How much effort was required to inspect the proposed quizzes and decide whether to keep or change them?"

Q3 - Mental Demand:
"How mentally demanding was it to read and evaluate the candidate quizzes?"

Q4 - Perceived Controllability:
"How much control did you feel you had over the final quiz? Was it mostly determined by your inputs, or did it feel unpredictable?"

Q5 - Temporal Demand:
"How time-pressured did you feel while composing the quiz?"

Q6 - Satisfaction:
"Overall, how satisfied are you with the final quiz you produced?"

Provide your responses in JSON format:
{{
  "accomplishment": <1-5>,
  "effort": <1-5>,
  "mental_demand": <1-5>,
  "controllability": <1-5>,
  "temporal_demand": <1-5>,
  "satisfaction": <1-5>,
  "reasoning": "brief explanation of your ratings (2-3 sentences)"
}}
"""

def build_composition_prompt(
    iteration: int,
    previous_quizzes: list = None,
    prompt_type: str = "detailed"
) -> str:
    """Build prompt for quiz composition iteration."""
    
    # System prompt
    if prompt_type == "detailed":
        system_prompt = PROMPT_TEMPLATE_COMPOSITION_DETAILED
    else:
        system_prompt = PROMPT_TEMPLATE_COMPOSITION_SIMPLE
    
    # Iteration context
    if iteration == 0:
        iteration_context = """This is your FIRST composition attempt.

Please specify your initial quiz parameters:
- Topic distribution (e.g., [0.3, 0.3, 0.4] for 30% algebra, 30% geometry, 40% calculus)
- Difficulty distribution (e.g., [0.2, 0.5, 0.3] for 20% easy, 50% medium, 30% hard)"""
        previous_results = ""
    else:
        iteration_context = f"""This is iteration {iteration + 1} of quiz composition."""
        
        if previous_quizzes:
            last_quiz = previous_quizzes[-1]
            previous_results = f"""
Previous quiz results:
- Number of questions: {last_quiz.get('num_mcqs', 0)}
- Match quality: {last_quiz.get('target_match', 0):.1%}
- Generation time: {last_quiz.get('generation_time_s', 0):.1f}s
- Topics: {last_quiz.get('topic_distribution', [])}
- Difficulty: {last_quiz.get('difficulty_distribution', [])}
"""
        else:
            previous_results = ""
    
    # Combine
    user_prompt = COMPOSITION_ITERATION_PROMPT.format(
        iteration_context=iteration_context,
        previous_results=previous_results
    )
    
    return system_prompt, user_prompt


def build_survey_prompt(trajectory_summary: str) -> tuple:
    system_prompt = PROMPT_TEMPLATE_SURVEY.format(
        trajectory_summary=trajectory_summary
    )
    
    user_prompt = SURVEY_QUESTIONS_PROMPT
    
    return system_prompt, user_prompt
