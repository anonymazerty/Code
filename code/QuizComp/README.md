# QuizComp: Adaptive Quiz Composition with RL and LLM-Based Evaluation

QuizComp is a research platform for **adaptive quiz composition** using Reinforcement Learning (RL). Teachers specify topic and difficulty preferences; an RL agent searches a universe of candidate quizzes and returns the best match. The platform includes a **user study interface** (Streamlit), a **FastAPI backend**, and an **LLM simulation framework** that replays the same composition task with large language models to compare synthetic and human teacher behavior.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Backend API (`app/`)](#backend-api)
5. [User Interface (`quizcomp_ui/`)](#user-interface)
6. [LLM Simulation Study (`quizcomp_llm_study/`)](#llm-simulation-study)
7. [Data & Models](#data--models)
8. [User Study & Survey](#user-study--survey)
9. [Results & Analysis](#results--analysis)
10. [Configuration Reference](#configuration-reference)
11. [Requirements](#requirements)

## Project Structure

```
datagems/
├── app/                              # FastAPI Backend
│   ├── main.py                       # Uvicorn entry point (port 8000)
│   ├── app.py                        # FastAPI app setup, CORS, routers
│   ├── agents/                       # RL Agents
│   │   ├── base_agent.py             # Base agent class
│   │   ├── dqn_agent.py              # Deep Q-Network
│   │   ├── a2c_agent.py              # Advantage Actor-Critic
│   │   ├── a3c_agent.py              # Async Advantage Actor-Critic
│   │   └── sarsa_agent.py            # SARSA
│   ├── environments/
│   │   └── custom_env.py             # Custom Gym environment
│   ├── routers/                      # API endpoint routers
│   │   ├── composition.py            # POST /compose/quiz
│   │   ├── generation.py             # POST /gen/quizzes
│   │   └── mcqs.py                   # POST /mcqs/by_ids
│   ├── schemas/                      # Pydantic request/response models
│   ├── services/                     # Business logic (generation, composition, MCQs)
│   ├── replay_buffers/               # Experience replay (normal, prioritized)
│   └── utils/                        # Logging, utilities
│
├── quizcomp_ui/                      # Streamlit User Study Interface
│   ├── str_app.py                    # Main Streamlit app
│   ├── study_ui.py                   # Study session flow management
│   ├── ui_components.py              # Quiz rendering components (LaTeX)
│   ├── db.py                         # SQLAlchemy models (SQLite)
│   ├── auth.py                       # User authentication (bcrypt)
│   ├── config.py                     # Topics, difficulty levels, API URL
│   ├── completion_codes_allocator.py # Prolific completion codes
│   └── migrate_add_q7.py            # DB migration script
│
├── quizcomp_llm_study/               # LLM Simulation Research
│   ├── config.py                     # LLM provider, model, parameters
│   ├── teacher_persona.py            # TeacherPersona data structure
│   ├── db_saver.py                   # Results DB persistence
│   ├── clean_and_extract.py          # Extract personas from user study DB
│   │
│   ├── # Interactive Simulation (e2eAgent)
│   ├── interactive_prompt.py         # Prompt routing to 3 variants
│   ├── interactive_llm_teacher_agent.py  # Agent with real API calls
│   ├── interactive_simulation_engine.py  # Orchestration engine
│   ├── run_interactive_simulation.py     # CLI entry point
│   │
│   ├── # Survey-Only Simulation (JudgeAgent)
│   ├── prompt.py                     # Survey prompt templates
│   ├── llm_teacher_agent.py          # Survey-only agent
│   ├── simulation_engine.py          # Survey-only orchestration
│   ├── survey_only_simulation.py     # CLI entry point
│   │
│   ├── prompts/                      # Prompt template variants
│   │   ├── calibrated_prompt.py      # Calibrated: full behavior spec
│   │   ├── role_anchored.py          # Role-Anchored: strong identity
│   │   └── task_constrained.py       # Task-Constrained: minimal rules
│   │
│   ├── scripts/                      # Post-hoc analysis scripts
│   │   ├── extract_tokens_and_costs.py
│   │   ├── extract_judge_tokens_and_costs.py
│   │   ├── simulate_io_costs.py
│   │   ├── export_e2e_agent_csvs.py
│   │   ├── export_judge_agent_csvs.py
│   │   ├── compute_alignment_scores.py
│   │   ├── compute_group_medians.py
│   │   ├── plot_e2e_agent_histograms.py
│   │   ├── plot_judge_agent_histogram.py
│   │   ├── judge_agent_statistical_tests.py
│   │   └── add_job_role_and_analyse.py
│   │
│   ├── Results/                      # All results and analysis output
│   │   ├── e2eAgent/Results_as_csvs/ # Per-run CSVs, tokens, costs
│   │   ├── JudgeAgent/Results_as_csvs/
│   │   └── dbs/                      # Source SQLite databases
│   │
│   ├── pricing_config.json           # LLM pricing rates
│   └── teacher_personas.json         # 65 extracted personas
│
├── data/                             # MCQ dataset + generated universes
│   └── math.csv                      # math MCQs 
├── models/                           # Pre-trained RL models (DQN)
├── output_quizzes/                   # Quiz outputs from user sessions
├── quizcomp_study.sqlite             # User study database
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone <repo-url>
cd datagems
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running the System

**1. Start the Backend API**

```bash
python -m app.main
# API → http://localhost:8000
# Docs → http://localhost:8000/docs
```

**2. Start the User Interface**

```bash
cd quizcomp_ui
streamlit run str_app.py
# UI → http://localhost:8501
```

**3. Run LLM Simulations** (requires API keys)

```bash
# Interactive simulation (e2eAgent)
export OPENAI_API_KEY="sk-..."
python -m quizcomp_llm_study.run_interactive_simulation

# Survey-only simulation (JudgeAgent)
python -m quizcomp_llm_study.survey_only_simulation
```

---

## Backend API

The FastAPI backend provides three endpoints for quiz generation and composition.

### `POST /gen/quizzes` — Generate Quiz Universe

Creates a universe of candidate quizzes by sampling combinations of MCQs.

| Parameter | Type | Description |
|---|---|---|
| `MCQs` | List[str] | Paths to MCQ CSV files |
| `numQuizzes` | int | Number of quizzes to generate |
| `numMCQs` | int | Questions per quiz |
| `topicMode` | int | 0 = same topic, 1 = different topics |
| `levelMode` | int | 0 = same level, 1 = different levels |

Returns a `RequestID` (UUID) used for subsequent composition calls.

### `POST /compose/quiz` — Compose Best-Match Quiz

Uses an RL agent (DQN by default) to search the quiz universe for the best match to teacher preferences.

| Parameter | Type | Description |
|---|---|---|
| `dataUUID` | UUID | Quiz universe ID from `/gen/quizzes` |
| `teacherTopic` | List[float] | Target topic distribution (sums to 1.0) |
| `teacherLevel` | List[float] | Target difficulty distribution (sums to 1.0) |
| `pathToModel` | str | Path to pre-trained RL model |
| `alfaValue` | float | Weight: 0 = optimize difficulty, 1 = optimize topics |
| `startQuizId` | int? | Starting quiz ID for improve mode (null = fresh) |

Returns the best quiz with `targetMatch` (0–1 match quality).

### `POST /mcqs/by_ids` — Retrieve MCQ Details

Fetches full question content for a list of MCQ IDs.

### RL Agents

Four RL agent implementations search the quiz universe:

| Agent | File | Algorithm |
|---|---|---|
| DQN | `agents/dqn_agent.py` | Deep Q-Network with experience replay |
| A2C | `agents/a2c_agent.py` | Advantage Actor-Critic |
| A3C | `agents/a3c_agent.py` | Asynchronous Advantage Actor-Critic |
| SARSA | `agents/sarsa_agent.py` | On-policy TD control |

The custom Gym environment (`environments/custom_env.py`) models quiz composition as a sequential decision problem where the agent navigates through the quiz universe to maximize match quality with teacher preferences.

---

## User Interface

The Streamlit-based UI (`quizcomp_ui/`) manages the complete user study workflow.

### Study Flow

1. **Authentication** — Login/register with Prolific PID integration
2. **Consent** — Informed consent before participation
3. **Quiz Composition** — Interactive loop:
   - Select number of questions (3–30)
   - Set topic percentages across 10 math topics
   - Set difficulty level distribution (6 levels)
   - Review generated quiz (with LaTeX rendering)
   - **Accept**, **Improve** (same params), or **Fresh** (new params)
4. **Post-Study Survey** — Likert-scale questions Q1–Q6
5. **Completion Code** — Prolific completion code issued


### Database Schema (SQLite)

| Table | Purpose |
|---|---|
| `users` | User accounts (hashed passwords) |
| `study_sessions` | Session tracking, Prolific IDs, consent, duration |
| `event_logs` | Page views, clicks, API calls |
| `compose_attempts` | Parameters, results, coverage vectors, response times |
| `survey_responses` | Q1–Q6 Likert responses + Q7 (reuse) + comments |

---

## LLM Simulation Study

The LLM simulation framework (`quizcomp_llm_study/`) tests whether LLMs can replicate human teacher behavior in the quiz composition task. Two distinct agent architectures are evaluated.

### Two Agent Types

#### e2eAgent (Interactive / End-to-End)

The LLM **actually uses** the composition system via real API calls:

1. Receives only the initial parameters from a real teacher's first attempt
2. Calls the real `/compose/quiz` API
3. Analyzes the returned quiz (match quality, topic/difficulty coverage)
4. Decides: **Accept**, **Improve**, or **Fresh**
5. Iterates until satisfied or max iterations reached
6. Completes the Q1–Q6 survey based on its own experience

This tests **behavioral simulation**: can the LLM act like a human teacher?

#### JudgeAgent (Survey-Only)

The LLM receives the complete interaction history of a real teacher session and answers the survey without interacting with the system. This tests **interpretive simulation**: can the LLM infer how a human would feel from their history?

### Three Prompt Strategies

Each e2eAgent simulation is run with three prompt variants that control how much behavioral guidance the LLM receives:

| Prompt Type | Description | Key Characteristics |
|---|---|---|
| **Calibrated** | Full behavior specification | Detailed match-quality interpretation, content checks, explicit decision rules, iteration limits |
| **Role-Anchored** | Strong teacher identity | Heavy emphasis on "you are a REAL MATH TEACHER", pragmatic behavior, accept quickly |
| **Task-Constrained** | Minimal task rules | Light constraints ("good enough" at 70–90%), practical decision-making, time pressure |

### Three LLM Models

| Model | Provider | Use Case |
|---|---|---|
| GPT-5 (`gpt-5-2025-08-07`) | OpenAI (direct API) | State-of-the-art proprietary |
| LLaMA 3 8B (`llama3.1:8b`) | OpenRouter / Ollama | Open-source small |
| Mistral Small 24B (`mistralai/mistral-small-3.2-24b-instruct`) | OpenRouter | Open-source medium |

### Simulation Parameters

- **65 teacher personas** extracted from real user study data
- **5 runs per persona** per configuration
- **325 simulations** per (model × prompt type) combination
- **Max 10 iterations** per session (real user)
- **10 questions** per quiz

### Running Simulations

```bash
# Set provider and API key
export OPENAI_API_KEY="sk-..."
# or
export OPENROUTER_API_KEY="..."

# Interactive e2eAgent simulation
python -m quizcomp_llm_study.run_interactive_simulation

# With specific options
python -m quizcomp_llm_study.run_interactive_simulation \
  --test \
  --filter-persona-ids 0 1 2

# Survey-only JudgeAgent simulation
python -m quizcomp_llm_study.survey_only_simulation
```

---

## Data & Models

### MCQ Dataset

`data/math.csv` contains 1,128 multiple-choice math questions that was deleted for privacy.

| Column | Description |
|---|---|
| `id` | Unique integer identifier |
| `question` | Question text (may include LaTeX) |
| `option_a` – `option_d` | Four answer choices |
| `correct_option` | Correct answer (A/B/C/D) |
| `topic_name` | One of 10 math topics |
| `difficulty` | Difficulty level (1–6) |

### Pre-trained RL Models

Eight DQN models in `models/`, trained on different dataset/configuration combinations:

| Model | Training Data | Tier |
|---|---|---|
| `dqn_t1_math_r2` | Math | 1 |
| `dqn_t4_math_r2` | Math (default) | 4 |
| `dqn_t1_medical_r2` | Medical | 1 |
| `dqn_t4_medical_r2` | Medical | 4 |
| `dqn_t1_uniform_r2` | Uniform | 1 |
| `dqn_t4_uniform_r2` | Uniform | 4 |
| `dqn_t1_disdiff_r2` | Difficulty-diverse | 1 |
| `dqn_t1_distop_r2` | Topic-diverse | 1 |

Tier variants (`t1`, `t4`) differ in training hyperparameters. The `_r2` suffix indicates reward function version 2.

---

## User Study & Survey

### Participants

65 teachers recruited via Prolific, each completing one full composition session. Their interaction data (initial parameters, iteration choices, timings) serves as ground truth for LLM simulation comparison.

### Survey Questions (Q1–Q6, Likert 1–5)

| # | Dimension | Question |
|---|---|---|
| Q1 | Accomplishment | How successful do you feel you were in building a quiz that matches your needs? |
| Q2 | Effort | How much effort was required to inspect the proposed quizzes and decide whether to keep or change them? |
| Q3 | Mental Demand | How mentally demanding was it to read and evaluate the candidate quizzes? |
| Q4 | Controllability | How much control did you feel you had over the final quiz? |
| Q5 | Temporal Demand | How time-pressured did you feel while composing the quiz? |
| Q6 | Satisfaction | Overall, how satisfied are you with the final quiz you produced? |

---

## Results & Analysis

### Results Structure

```
quizcomp_llm_study/Results/
├── e2eAgent/
│   └── Results_as_csvs/
│       ├── Calibrated/{GPT5,llama8B,Mistral}/    # Per-run survey + token CSVs
│       ├── Role-Anchored/{GPT5,llama8B,Mistral}/
│       ├── Task-Constrained/{GPT5,llama8B,Mistral}/
│       ├── tokens_summary.csv                      # Aggregate tokens
│       ├── cost_summary.csv                        # Cost estimates
│       └── cost_simulation_summary.csv             # I/O simulated costs
├── JudgeAgent/
│   └── Results_as_csvs/
│       ├── {GPT5,llama8B,Mistral}/                # Per-run survey + token CSVs
│       ├── tokens_summary.csv
│       ├── cost_summary.csv
│       └── cost_simulation_summary.csv
└── dbs/                                            # Source SQLite databases
    ├── e2eAgent dbs/
    ├── JudgeAgent dbs/
    └── Crowd db/
```

### Analysis Scripts

All scripts are in `quizcomp_llm_study/scripts/`:

| Script | Purpose |
|---|---|
| `export_e2e_agent_csvs.py` | Export e2eAgent results from DBs to CSVs |
| `export_judge_agent_csvs.py` | Export JudgeAgent results from DBs to CSVs |
| `extract_tokens_and_costs.py` | Token counts and cost estimates (e2eAgent) |
| `extract_judge_tokens_and_costs.py` | Token counts and cost estimates (JudgeAgent) |
| `compute_alignment_scores.py` | LLM vs. human survey response alignment |
| `compute_group_medians.py` | Group-level median statistics |
| `add_job_role_and_analyse.py` | Role-based sub-group analysis |
| `judge_agent_statistical_tests.py` | Statistical significance tests |
| `plot_e2e_agent_histograms.py` | Response distribution visualizations |
| `plot_judge_agent_histogram.py` | JudgeAgent visualizations |

### LLM Cost Summary

Pricing per 1M tokens (see `pricing_config.json`):

| Model | Input $/1M | Output $/1M |
|---|---:|---:|
| GPT-5 | $2.50 | $20.00 |
| LLaMA 3 8B | $0.036 | $0.038 |
| Mistral Small 24B | $0.053 | $0.095 |

---

## Configuration Reference

### Backend (`app/`)

No environment variables required for basic operation. Models and data paths are configured in code. Default port: 8000.

### Frontend (`quizcomp_ui/`)

| Variable | Default | Description |
|---|---|---|
| `QUIZCOMP_API_BASE` | `http://127.0.0.1:8000` | Backend API URL |
| `QUIZCOMP_DB_PATH` | `quizcomp_study.sqlite` | SQLite database file |

### LLM Study (`quizcomp_llm_study/config.py`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `"openai"` | Provider: `"openai"`, `"openrouter"` |
| `OPENAI_API_KEY` | env | OpenAI API key |
| `OPENROUTER_API_KEY` | env | OpenRouter API key |
| `MAX_ITERATIONS` | 10 | Max composition iterations per session |
| `QUESTIONS_PER_QUIZ` | 10 | Target questions per quiz |
| `PROMPT_TYPE` | `"detailed"` | Prompt template variant |

---

## Requirements

Core dependencies (see `requirements.txt`):

- **API**: FastAPI, Uvicorn, Pydantic
- **UI**: Streamlit
- **RL**: PyTorch, Gym (0.26.2)
- **Data**: Pandas, NumPy, SciPy, scikit-learn
- **Database**: SQLAlchemy, SQLite
- **Auth**: passlib, bcrypt
- **Visualization**: Matplotlib, Seaborn, Plotly
- **LLM**: OpenAI Python SDK

---

## License

Research use only.

