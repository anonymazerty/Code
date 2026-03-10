# Interactive LLM Teacher Simulation - Quick Summary

## What Is This?

A system that tests whether Large Language Models (LLMs) can simulate real human behavior by actually **using** a quiz composition system, not just analyzing historical data.

## The Problem We're Solving

**Previous approach (Survey-Only):**
- Give LLM complete history of a user's interactions
- Ask LLM to answer survey questions based on that history
- **Issue**: LLM is just interpreting someone else's experience, not having their own

**New approach (Interactive):**
- Give LLM only initial parameters (like a real user would start with)
- Let LLM actually interact with the composition system via API calls
- LLM makes real decisions: accept quiz, try to improve it, or change parameters
- LLM answers survey based on their **own** experience
- **Advantage**: Tests actual behavioral simulation, not just interpretation

## How It Works

### 1. Setup (from real users)

Extract initial parameters from real teachers' first composition attempt:
- Topic distribution: `[0.3, 0.3, 0.4]` (30% algebra, 30% geometry, 40% calculus)
- Difficulty distribution: `[0.2, 0.5, 0.3]` (20% easy, 50% medium, 30% hard)
- Number of questions: `10`

### 2. Interactive Loop

The LLM agent then uses the system:

```
Iteration 1:
  → Call API with initial parameters
  ← Receive quiz (ID=123, Match=85%)
  → Agent decides: "Try to improve"
  
Iteration 2:
  → Call API (improve mode, start from quiz 123)
  ← Receive quiz (ID=156, Match=91%)
  → Agent decides: "Good enough, accept!"
```

### 3. Survey

Agent answers Q1-Q6 based on their actual experience:
- Accomplishment: 4/5
- Effort: 3/5
- Mental Demand: 3/5
- Controllability: 4/5
- Temporal Demand: 2/5
- Satisfaction: 4/5

### 4. Comparison

Compare LLM behavior and responses with real users:
- Do they iterate similar numbers of times?
- Do they accept at similar match quality thresholds?
- Do their survey responses align with their behavior?
- Do their responses match real users' responses?

## Two Composition Modes

**Fresh Mode:**
- Change topic/difficulty parameters
- Start search from scratch
- Used when agent wants different content

**Improve Mode:**
- Keep same parameters
- Resume search from last quiz
- Used when agent wants better match with same parameters

## Human-Like Behavior Prompting

Critical component: prompts explicitly instruct LLM to behave like a **real teacher**, not an AI optimizer:

 **DO:**
- Accept "good enough" quizzes (80%+ match is fine)
- Consider time pressure (iterating costs time)
- Get tired after a few attempts
- Make pragmatic, imperfect decisions

 **DON'T:**
- Seek mathematical perfection
- Iterate 10+ times for marginal improvements
- Be overly systematic or rational
- Give verbose AI-like explanations

## Files Created

1. **`interactive_llm_teacher_agent.py`**: Agent that actually uses the composition API
2. **`interactive_simulation_engine.py`**: Orchestrates simulations across multiple personas
3. **`interactive_prompt.py`**: Prompts for human-like interactive behavior
4. **`run_interactive_simulation.py`**: Main entry point / CLI
5. **`INTERACTIVE_METHODOLOGY.md`**: Complete methodology documentation
6. Updated `db_saver.py`: Save interactive simulation results
7. Updated `README.md`: Documentation

## Quick Usage

### Test Run (Custom Parameters)
```bash
python -m quizcomp_llm_study.run_interactive_simulation --test
```

### Run All Personas
```bash
python -m quizcomp_llm_study.run_interactive_simulation
```

### Run Specific Persona
```bash
python -m quizcomp_llm_study.run_interactive_simulation --persona-id 0
```

### Prerequisites

1. QuizComp API server running:
   ```bash
   python -m app.main  # Runs on http://localhost:8000
   ```

2. LLM API key in `.env`:
   ```
   OPENROUTER_API_KEY=your-key-here
   # or
   OPENAI_API_KEY=your-key-here
   ```

## Results

All results saved to:
- **Database**: `llm_results.db` (SQLite)
  - Table: `simulations` (with `prompt_type='interactive'`)
  - Table: `composition_attempts` (actual API interactions)
  - Table: `llm_survey_responses` (Q1-Q6 ratings)
  - Table: `teacher_personas` (real user data for comparison)

- **JSON files**: `interactive_results/`
  - Complete interaction histories
  - Agent decisions and reasoning
  - Survey responses

## Research Value

1. **Behavioral Fidelity**: Tests if LLMs can actually behave like humans, not just understand behavior
2. **Survey Validity**: Tests if LLM survey responses match their actual experiences
3. **Persona Modeling**: Evaluates LLMs as human behavioral models
4. **UX Evaluation**: Identifies UX issues through simulated usage patterns
5. **System Validation**: Validates composition system through realistic simulated users

## Key Insights

### What We Can Learn

1. **Behavioral Patterns**:
   - How many iterations do LLMs do vs. real users?
   - When do LLMs accept quizzes (match quality threshold)?
   - How do LLMs use fresh vs. improve modes?

2. **Decision Making**:
   - Are LLM decisions pragmatic or over-optimizing?
   - Do LLMs show human-like "satisficing" (good enough)?
   - Do LLMs respond to time pressure?

3. **Survey Consistency**:
   - Do LLM survey responses align with their behavior?
   - E.g., Low "effort" ↔ Few iterations?
   - E.g., High "satisfaction" ↔ High match quality?

4. **Human Similarity**:
   - How well do LLM behaviors match real user behaviors?
   - How well do LLM survey responses match real user responses?
   - Can LLMs serve as realistic behavioral models?

## Example Output

```

INTERACTIVE SIMULATION for Persona 0 (Run 0)


Initial Parameters:
  Topics: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
  Difficulty: [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
  Num MCQs: 10

--- Iteration 1 (fresh mode) ---
✓ Quiz generated: ID=342, Match=87%, Time=3.2s
  Decision: compose
  Mode: improve
  Reasoning: Match is good but let me try once more...

--- Iteration 2 (improve mode) ---
✓ Quiz generated: ID=451, Match=91%, Time=2.8s
  Decision: accept
  Reasoning: This is good enough, I need to move on

✓ Agent accepted quiz after 2 iterations

Survey responses:
  accomplishment: 4/5
  effort: 3/5
  mental_demand: 3/5
  controllability: 4/5
  temporal_demand: 2/5
  satisfaction: 4/5

Real Teacher Survey Responses (for comparison):
  accomplishment: 5/5
  effort: 2/5
  mental_demand: 2/5
  controllability: 5/5
  temporal_demand: 1/5
  satisfaction: 5/5


SIMULATION COMPLETE
Duration: 18.3s | Iterations: 2 | Final Match: 91%

```

## Next Steps

1. **Run simulations** on all personas
2. **Analyze results**:
   - Behavioral patterns (iterations, modes, thresholds)
   - Survey response distributions
   - Consistency between behavior and surveys
   - Similarity to real users
3. **Compare models**: Test different LLMs (GPT-4, Claude, etc.)
4. **Refine prompts**: Improve human-like behavior
5. **Publish findings**: Behavioral simulation paper

## Comparison Table

| Aspect | Survey-Only | Interactive |
|--------|------------|-------------|
| **Input** | Full interaction history | Only initial parameters |
| **LLM Task** | Answer survey questions | Use system + answer survey |
| **API Calls** | None (simulated) | Real composition API calls |
| **Behavior Tested** | No | Yes (accept/improve/fresh) |
| **Survey Basis** | Others' experience | Own experience |
| **Realism** | Interpretation | Actual usage |
| **Research Value** | Survey prediction | Behavioral modeling |
| **Requirements** | Database only | Database + API server |

