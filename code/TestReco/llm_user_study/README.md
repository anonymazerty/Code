# LLM User Study — Learning Simulation Framework

This directory contains the complete LLM-based simulation system for recreating student learning experiences.

## Overview

The system simulates how LLMs behave as learners taking the same learning journey as real students. It uses personalized recommendation to guide learning through adaptive question sequences while an LLM plays the role of the student. Three models are supported: **LLaMA-3-8B**, **GPT-5**, and **Mistral-24B-Instruct**. Simulations are run across three prompt strategies for learning: `detailed` (Calibrated), `general` (Task-Constrained), `simple` (Role-Anchored), and one prompting strategy for qualitative study `survey` (Survey-only).

## Directory Structure

```
llm_user_study/
├── config.py                  # Central configuration
├── run_simulation.py          # Main CLI for e2eAgent simulations
├── simulation_engine.py       # Core simulation engine
├── llm_student_agent.py       # LLM agent (answers questions + surveys)
├── persona_builder.py         # Builds personas from real user DB
├── prompt.py                  # All prompt templates
├── db_saver.py                # Saves results to SQLite
├── survey_only_simulation.py  # AgentJudge mode (survey-only)
├── personas.json              # 55 personas from real learners
├── results.db                 # SQLite database (symlink)
├── results/                   # Generated outputs
│   ├── JudgeAgent/            #   AgentJudge results
│   ├── e2eAgent/              #   e2eAgent results
│   └── alignment_results/     #   Bootstrap alignment tables
└── scripts/
    ├── bootstrapping.py       # Bootstrap CIs, compatibility, alignment
    └── export_results/        # Plotting & export scripts
```

## Core Components

### Main Execution Scripts

#### `run_simulation.py`
Main CLI entry point for **e2eAgent** simulations.

- **Class**: `LLMStudyPipeline` — loads personas, skips already-completed runs (by `simulated_session_id` + `prompt_type` + `model_name`), runs simulations.
- **Usage**:
  ```bash
  python -m llm_user_study.run_simulation --prompt-type detailed
  python -m llm_user_study.run_simulation --prompt-type simple --model llama-3-8b
  python -m llm_user_study.run_simulation --prompt-type general --persona-ids 0,1,2
  ```
- **CLI Arguments**:
  | Arg | Default | Description |
  |---|---|---|
  | `--output-dir` | `llm_user_study` | Base output directory |
  | `--personas-path` | `llm_user_study/personas.json` | Personas JSON file |
  | `--persona-ids` | None | Comma-separated IDs to filter |
  | `--prompt-type` | `detailed` | `detailed` / `general` / `simple` / `survey` |
  | `--model` | None | Override model for all education tiers |
  | `--force` | `False` | Re-run even if already in DB |
  | `--runs` | `1` | Runs per persona |

#### `survey_only_simulation.py`
**AgentJudge** mode — gives the LLM a real learner's complete trajectory and asks it to complete the NASA-TLX survey without answering any questions.

- **Key Functions**:
  - `load_real_learner_trajectory()` — loads steps + attempts from DB
  - `build_trajectory_string()` — formats trajectory with per-question details
  - `simulate_survey_only()` — full pipeline: persona → trajectory → survey → DB
  - `run_test_retest()` — test-retest reliability mode (9 profiles/level × N runs)
- **Usage**:
  ```bash
  python -m llm_user_study.survey_only_simulation --session-id 45
  python -m llm_user_study.survey_only_simulation --batch --model llama-3-8b --runs 5
  python -m llm_user_study.survey_only_simulation --test-retest --model llama-3-8b --runs 5
  ```
- **CLI Arguments**:
  | Arg | Description |
  |---|---|
  | `--session-id` | Single session |
  | `--session-ids` | Comma-separated list |
  | `--batch` | All completed sessions |
  | `--model` | Model name override |
  | `--runs` | Runs per session |
  | `--start-run` | Starting run number |
  | `--test-retest` | Test-retest mode (9 profiles/level) |
  | `--profiles-per-level` | Profiles per level (default 9) |

### Configuration

#### `config.py`
Central configuration for all simulations.

| Constant | Value |
|---|---|
| `DB_PATH` | env `REALUSER_DB_PATH` or `./TestReco/results.db` |
| `TOPIC` | `"Fundamental Mathematics"` |
| `BENCHMARK` | `"math_bench"` |
| `ORCHESTRATOR_TYPE` | `"tool_call"` |
| `MODEL_NAME` | `"chatgpt-4o-mini"` (orchestrator LLM) |
| `SIMULATIONS_PER_PERSONA` | `5` |
| `PROMPT_TYPE` | `"general"` |
| `PRETEST_N` | `12` |
| `STEPS_N` | `10` |
| `QUESTIONS_PER_STEP` | `5` |
| `OBJECTIVES` | `["performance", "gap", "aptitude"]` |
| `MODEL_TIERS` | All levels → `"gpt-5-2025-08-07"` |

- **`get_model_for_persona(education_level, persona_index)`** — round-robin model selection from `MODEL_TIERS`.

#### `personas.json`
55 personas extracted from real user study sessions.

| Field | Description |
|---|---|
| `persona_id` | Unique identifier (0–54) |
| `session_id` | Corresponding real user session ID |
| `education_level` | `"high_school"` (9), `"undergraduate"` (23), `"graduate"` (23) |
| `model_name` | Default model (all `"gpt-5-2025-08-07"`) |
| `qualification_score` | Pretest score (1–12) |
| `pre_qualification_score` | Always `0.4` (learning-phase start mastery) |
| `final_mastery` | Real learner's final mastery |
| `pretest_questions` | 12 question indices |
| `pretest_correctness` | 12 booleans |
| `pretest_chosen_options` | 12 chosen option indices |

### Simulation Engine

#### `simulation_engine.py`
Core engine for running full TestReco learning sessions with LLM personas.

- **`SimulationResult`** (dataclass) — captures: persona_id, run_number, session_id, pretest/final mastery, mastery/accuracy/rewards per step, learning_steps_data, survey_responses, orchestrator_calls, llm_student_calls, duration, prompt_type.
- **`TestRecoSimulator`** — initializes `RealUserEngine`, loads questions/difficulties, sets up orchestrator (`tool_call`, `reflection`, or `context`). Three-phase simulation:
  1. **Pretest** — uses real user's pretest data (no LLM answering)
  2. **Learning loop** — 10 steps: orchestrator selects strategy → questions selected → LLM answers each → env updated (mastery, aptitude, gap)
  3. **Survey** — builds full trajectory string, LLM completes NASA-TLX survey
- **`MultipleRunsRunner`** — batch runner: iterates personas, runs N simulations each, saves via `SimulationDatabaseSaver`.
- Tracks **token usage** and **latency** for both orchestrator calls and LLM student calls.

#### `llm_student_agent.py`
LLM agent that simulates student behavior — answering MCQs and completing surveys.

- **Model support** via `model_mapping` dict: `gpt-5-2025-08-07`, `chatgpt-4o`, `chatgpt-4o-mini`, `claude-3.7-sonnet-thinking`, `llama-3-8b`, `llama-3-70b`, `gemma-2-9b-it`, `mistral-24b-instruct`
- Uses direct **OpenAI API** when `OPENAI_API_KEY` is set and model starts with `openai/`; otherwise falls back to **OpenRouter**.
- **`answer_question(question, difficulty, context)`** → `(chosen_idx, response_time_ms, metadata)`. Builds prompt with mastery level and education level; simulates realistic response time (3–12s with difficulty factor + noise).
- **`complete_survey(session_data)`** — uses `PROMPT_TEMPLATE_SURVEY` as system prompt. Expects JSON output with 7 NASA-TLX items rated 1–5.
- **`_validate_survey(survey_data, mastery_change)`** — clips values to 1–5, maps `controllability` → `perceived_controllability`, sets `would_use_again = 1`.
- **Survey dimensions**: accomplishment, effort_required, mental_demand, perceived_controllability, temporal_demand, frustration, trust, would_use_again.

### Persona & Prompt Management

#### `persona_builder.py`
Builds `LLMPersona` objects from the real user study database.

- **`LLMPersona`** (dataclass):
  - `get_system_prompt(prompt_type, learning_trajectory)` — dispatches to the appropriate template
  - `_build_pretest_display()` — formats pretest into "CORRECTLY answered" / "INCORRECTLY answered" sections with full question text, options, and markers (`← YOUR CHOICE`, `✓ CORRECT`)
- **`PersonaBuilder`**:
  - `get_completed_sessions()` — sessions with 10 learning steps + survey + not LLM-simulated
  - `build_persona(session_id, persona_id)` — queries `study_sessions`, `users`, `prolific_demographics`, `attempts`
  - `build_all_personas()` / `save_personas()` — batch build and save to JSON

#### `prompt.py`
Four prompt templates:

| Template | Prompt Strategy | Description |
|---|---|---|
| `PROMPT_TEMPLATE_DETAILED` | Calibrated | Difficulty scale `d [0.0–1.0]`, mastery `m`, guardrails, education-specific distractor rules |
| `PROMPT_TEMPLATE_GENERAL` | Task-Constrained | Mastery + education level + pretest, instruction to not overperform |
| `simple_system_prompt` | Role-Anchored | Minimal: "You are roleplaying as a student. You are NOT an LLM." Pretest only |
| `PROMPT_TEMPLATE_SURVEY` | Survey-only | Full learning trajectory injected via `{learning_trajectory}`, honest reflection |

**`EDUCATION_EXPLANATIONS`** — per-level error pattern descriptions:
- **Graduate**: sophisticated distractors, subtle conceptual confusion
- **Undergraduate**: plausible distractors, incomplete understanding
- **High School**: any distractor, fundamental misunderstandings, random guessing

**Template variables**: `{education_level}`, `{education_explanations}`, `{initial_mastery}`, `{qualification_score}`, `{pretest_questions}`, `{learning_trajectory}`

### Database

#### `db_saver.py`
Saves LLM simulation results to the same SQLite schema as the real user study.

**`SimulationDatabaseSaver`** performs 8 save operations per simulation:
1. `_get_or_create_llm_user` — creates user `llm_{edu}_{persona_id}`
2. `_create_session` — inserts into `study_sessions` with `simulated_session_id`, `prompt_type`, `model_name`
3. `_save_pretest_attempts` — real user's pretest into `attempts` (phase=`pretest`)
4. `_save_learning_steps` — `learning_steps` (mastery_before/after, rewards) + `attempts` (phase=`learning`)
5. `_save_orchestrator_calls` — `orchestrator_calls` (strategy, latency, tokens)
6. `_save_survey` — `survey_responses` (7 Likert fields + would_use_again)
7. `_save_llm_student_calls` — `llm_student_calls` table (per-call token/latency, call_type = `question`/`survey`)
8. `_update_session_final` — updates `study_sessions` with `final_mastery` and `final_accuracy`

**Database tables written to**: `users`, `study_sessions`, `attempts`, `learning_steps`, `orchestrator_calls`, `survey_responses`, `llm_student_calls`

**Additional tables** (from `survey_only_simulation.py`): `survey_test_retest`, `llm_survey_predictions`

## Analysis & Export Scripts

Located in `scripts/`:

### `scripts/bootstrapping.py`
Bootstrap Compatibility:
- Bootstrap CIs for crowd median and IQR (2000 resamples)
- Agent–Crowd compatibility checks (✓/✗)
- Delta tables, Wilcoxon rank-sum tests
- Runs for both **e2eAgent** and **AgentJudge** modes, overall and per education level

### `scripts/export_results/`

| Script | Purpose |
|---|---|
| `export_csvs.py` | Export e2eAgent survey data to CSV (crowd + per-model per-run) |
| `export_judge_csvs.py` | Export AgentJudge survey data to CSV |
| `export_table_csv.py` | Export combined stats summary tables |
| `export_edu_delta_tables.py` | Per-education-level delta tables for AgentJudge |
| `export_e2e_edu_delta_tables.py` | Per-education-level delta tables for e2eAgent |
| `judge_agent_stats.py` | Compute AgentJudge summary statistics |
| `compute_e2e_quantitative.py` | Compute e2eAgent quantitative summary (participation, latency, performance, rewards, strategy distribution) |
| `compute_trajectory_rmse.py` | RMSE of LLM trajectories vs crowd |
| `plot_trajectories.py` | Learning trajectory plots (crowd vs LLM models) |
| `plot_survey_histograms.py` | Survey response histogram comparisons |
| `plot_crowd_trajectory.py` | Crowd-only trajectory plot |

## Workflow

### e2eAgent Simulation

1. **Run**:
   ```bash
   python -m llm_user_study.run_simulation --prompt-type detailed --model llama-3-8b --runs 5
   ```
   - Loads 55 personas from `personas.json`
   - Skips already-completed (checks `simulated_session_id` + `prompt_type` + `model_name`)
   - 3-phase simulation: pretest → 10 learning steps → survey
   - Results saved to `results.db`

2. **Export & Analyze**:
   ```bash
   python scripts/export_results/export_csvs.py
   python scripts/bootstrapping.py
   python scripts/export_results/plot_trajectories.py --agent e2e --prompt-type simple
   python scripts/export_results/plot_survey_histograms.py --agent e2e --prompt-type simple
   ```

### AgentJudge (Survey-Only)

```bash
python -m llm_user_study.survey_only_simulation --batch --model llama-3-8b --runs 5
```
- Loads each real learner's complete trajectory from DB
- LLM completes survey based on trajectory (no question answering)
- Results saved to `survey_test_retest` table

### Results Structure

```
results/
├── JudgeAgent/
│   ├── results_as_csv/        # Per-model CSV exports
│   ├── learning_trajectories/ # Crowd trajectory plots
│   ├── survey_histograms/     # Survey comparison histograms
│   └── stats/                 # Summary statistics
├── e2eAgent/
│   ├── results_as_csv/        # Per-prompt/model CSV exports
│   │   ├── calibrated/        #   (detailed prompt)
│   │   ├── task-constrained/  #   (general prompt)
│   │   └── role-anchored/     #   (simple prompt)
│   ├── learning_trajectories/ # Trajectory plots + CSVs
│   └── survey_histograms/     # Per-prompt survey histograms
└── alignment_results/
    ├── alignment_results.csv  # Full alignment analysis
    ├── wilcoxon_results.csv   # Wilcoxon test results
    ├── align_*.tex            # LaTeX alignment tables
    ├── compat_*.tex           # LaTeX compatibility tables
    ├── delta_*.tex            # LaTeX delta tables
    └── cross_mode_comparison.tex
```

## Configuration Examples

### Use different models per education level
```python
MODEL_TIERS = {
    "high_school": ["mistral-24b-instruct"],
    "undergraduate": ["llama-3-8b"],
    "graduate": ["gpt-5-2025-08-07"],
}
```

## Output Files
- `results.db` — SQLite database with all simulation + real user data
- `personas.json` — Persona definitions (read-only, generated by `PersonaBuilder`)
- `results/` — All exported CSVs, plots, and LaTeX tables

