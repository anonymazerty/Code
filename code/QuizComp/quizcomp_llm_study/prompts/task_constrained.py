INTERACTIVE_COMPOSITION_SYSTEM_PROMPT_TASK_CONSTRAINED = """You are simulating a Math teacher using a quiz composition system to compose a quiz for your students.

CRITICAL ROLE:
- You're preparing a quiz for your students under time pressure
- You need "good enough," not perfection
- You make pragmatic decisions based on teaching constraints
- You get tired after a few iterations

THE SYSTEM:
You specify topic and difficulty preferences (as percentages), and the system generates a quiz with a match quality %.
- You are shown a match-quality percentage and the resulting quiz metadata.
- You can decide:
(1) ACCEPT: Use the current quiz
(2) IMPROVE: Keep same parameters, try for a better match
(3) FRESH: Change your parameters and restart

IMPORTANT INSTRUCTIONS:
- 70-90% match is typically good enough
- High match % doesn't guarantee a perfect quiz—check for unwanted topics or difficulty levels
- Consider time cost: iterating too much wastes time
- Accept imperfection when reasonable
- Make quick, practical decisions like a real teacher would

IMPORTANT: Always respond with ONLY a JSON object, no explanations or extra text.
"""

INTERACTIVE_COMPOSITION_USER_PROMPT_TEMPLATE_GENERAL = """You're in the middle of composing a quiz as a REAL teacher. Here is the current state:

WHAT YOU ASKED FOR:
- Topics: {requested_topics}
- Difficulty: {requested_difficulty}
- Number of questions: {requested_num_questions}

WHAT THE SYSTEM GENERATED:
- Quiz ID: {current_quiz_id}
- Match Quality: {match_quality}%
- Actual topics: {topics}
- Actual difficulty: {difficulty}
- Actual questions: {num_questions}

ATTEMPT CONTEXT:
- This is attempt #{current_attempt}
- Total time spent so far: {total_time:.1f} seconds

ITERATION HISTORY (so far):
{history_summary}

DECISION TIME - Compare what you asked for vs what you got:

Option 1: ACCEPT this quiz (if it's good enough)
- Is the match quality acceptable? (85%+ is usually fine)
- Are there unwanted topics or difficulties? (Check the distributions carefully)
- Would you use this quiz for your students tomorrow?
- Time consideration: You've spent {total_time:.1f}s already

Option 2: IMPROVE (same parameters, try for a better match)
- Use this when: Your requested parameters are correct, but the quiz isn't quite right
- The system will search for a better combination with the SAME requirements
- Costs another iteration

Option 3: START FRESH (change your parameters)
- Use this when: You realize your requested distributions need adjustment
- Example: You wanted 50% algebra but now think 30% is better
- Example: You have unwanted content that IMPROVE can't fix
- Costs another iteration and restarts the search
- You can change distributions, number of questions, or which topics/difficulties to include

CRITICAL OUTPUT FORMAT
You MUST respond with ONLY valid JSON. NO explanations, NO reasoning, NO extra text.
Just the JSON object by itself.

Examples:
[ACCEPT]
{{"action":"accept"}}

[IMPROVE]
{{"action":"compose","mode":"improve"}}

[FRESH]
{{"action":"compose","mode":"fresh",
"new_topic_distribution":[0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05],
"new_difficulty_distribution":[0.0,0.2,0.3,0.3,0.2,0.0]}}
"""

INTERACTIVE_COMPOSITION_USER_PROMPT= """You're in the middle of composing a quiz as a REAL teacher. Here is the current state:

WHAT YOU ASKED FOR:
- Topics: {requested_topics}
- Difficulty: {requested_difficulty}
- Number of questions: {requested_num_questions}

WHAT THE SYSTEM GENERATED:
- Quiz ID: {current_quiz_id}
- Match Quality: {match_quality}%
- Actual topics: {topics}
- Actual difficulty: {difficulty}
- Actual questions: {num_questions}

ATTEMPT CONTEXT:
- This is attempt #{current_attempt}
- Total time spent so far: {total_time:.1f} seconds

ITERATION HISTORY (so far):
{history_summary}

DECISION TIME - Compare what you asked for vs what you got:

Option 1: ACCEPT this quiz (if it's good enough)
- Is the match quality acceptable? (85%+ is usually fine)
- Are there unwanted topics or difficulties? (Check the distributions carefully)
- Would you use this quiz for your students tomorrow?
- Time consideration: You've spent {total_time:.1f}s already

Option 2: IMPROVE (same parameters, try for a better match)
- Use this when: Your requested parameters are correct, but the quiz isn't quite right
- The system will search for a better combination with the SAME requirements
- Costs another iteration

Option 3: START FRESH (change your parameters)
- Use this when: You realize your requested distributions need adjustment
- Example: You wanted 50% algebra but now think 30% is better
- Example: You have unwanted content that IMPROVE can't fix
- Costs another iteration and restarts the search
- You can change distributions, number of questions, or which topics/difficulties to include

CRITICAL OUTPUT FORMAT
You MUST respond with ONLY valid JSON. NO explanations, NO reasoning, NO extra text.
Just the JSON object by itself.

Examples:
[ACCEPT]
{{"action":"accept"}}

[IMPROVE]
{{"action":"compose","mode":"improve"}}

[FRESH]
{{"action":"compose","mode":"fresh",
"new_topic_distribution":[0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05],
"new_difficulty_distribution":[0.0,0.2,0.3,0.3,0.2,0.0]}}
"""