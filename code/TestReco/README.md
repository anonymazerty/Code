# Adaptive Question Recommendation 

A framework for multi-objective adaptive question recommendation in educational contexts. This codebase covers three interconnected components: the **Orchestrator System**, the **Real User Study**, and the **LLM User Study** (simulation).

---

## Repository Structure

```
TestReco/
├── agents/                    # RL agents (SARSA, A2C, PPO)
├── benchmarks/                # Question banks
├── configs/                   # JSON schemas for orchestrator tools
├── envs/                      # Educational environment 
├── generators/                # LLM model factory and wrappers
├── orchestrator/              # LLM orchestrator logic 
├── real_user_study/           # Streamlit web app for the human study
├── llm_user_study/            # LLM-as-student simulation framework
├── reward_handlers/           # Reward processing 
├── results/                   # Trained RL policies (required at runtime)
├── tools/                     # Policy factory and wrappers
├── utils/                     # Evaluation and training utilities
├── results.db                 # Shared SQLite database (real + simulated data)
├── main.py                    # Standalone orchestrator evaluator
├── train_evaluate_policy.py   # Train individual RL policies
├── viz.py                     # Analysis and visualization
└── Final_Requirements.txt     # Exact Python environment for deployment
```

---

## 1. Orchestrator System

The orchestrator is the core reasoning engine. At each learning step it decides **which trained RL policy** to apply, based on the learner's current state (mastery, gap, aptitude).

### Three Variants

| Variant | File | Description |
|---|---|---|
| **Tool-Call** (used in study) | `orchestrator/tool_call_orchestrator.py` | Calls policies as tools to simulate outcomes before deciding |
| **Context-Based** | `orchestrator/context_based_orchestrator.py` | Single-pass decision using full context |
| **Reflection-Based** | `orchestrator/reflection_based_orchestrator.py` | Iterative self-critique before committing |

### Standalone Evaluation

```bash
# Run orchestrator with trained policies
python3 main.py --orchestrator_model mistral-24b-instruct \
               --policy_dir results/ \
               --objectives "performance" "gap" "aptitude" \
               --episodes 50
```

### Training RL Policies

```bash
python3 train_evaluate_policy.py --agent ppo --objectives "performance" --train_episodes 10000
python3 train_evaluate_policy.py --agent a2c --objectives "gap"         --train_episodes 10000
python3 train_evaluate_policy.py --agent ppo --objectives "aptitude"    --train_episodes 10000
```

### Visualizations

```bash
# Mastery progression, radar plots, per-objective reward
python3 viz.py --config_file policy_comparison_base.json

# Pearson correlation scatter (aptitude vs gap under a performance policy)
python3 viz.py --config_file policy_comparison_base.json \
               --correlation_scatter_plot --optimized_obj performance

# Flow-zone alignment plot
python3 viz.py --config_file policy_comparison_base.json --flow_zone_plot

# Inference time, policy switching, and usage bar charts
python3 viz.py --config_file policy_comparison_all_orchestrators.json \
               --all_orchestrator_analysis
```

---

## 2. Real User Study (`real_user_study/`)

A **Streamlit web application** used by human participants recruited via Prolific.

### Participant Flow

1. Landing page
2. Login / registration
3. Consent form
4. Pretest (12 qualification questions, math benchmark)
5. Eligibility check (mastery threshold)
6. Adaptive learning phase — 10 steps × 5 questions, orchestrator-driven
7. NASA-TLX style post-study survey (7 Likert items, 1–5)
8. Completion page with Prolific code

### Run Locally

```bash
python -m streamlit run real_user_study/app.py --server.runOnSave=true
```

### Key Files

| File | Purpose |
|---|---|
| `app.py` | Entry point — all UI phases, session state, orchestrator calls |
| `db.py` | SQLAlchemy models and SQLite schema |
| `live_session.py` | `RealUserEngine` — drives the learning loop |
| `loader.py` | Loads benchmark questions and difficulties |
| `initial_estimation.py` | Pretest sampling and initial mastery estimation |
| `ui_components.py` | Reusable UI blocks (Likert, consent, survey) |
| `settings.py` | Study parameters (topic, steps, questions per step) |

### Database Tables Written

- `users` — participant accounts
- `study_sessions` — one row per session (model, status, mastery)
- `attempts` — per-question answer records (pretest + learning)
- `learning_steps` — per-step mastery/accuracy/rewards
- `orchestrator_calls` — LLM token usage + strategy selected
- `survey_responses` — 7 Likert items + would_use_again

### Environment Variables Required

```
OPENROUTER_API_KEY=...   # For orchestrator LLM calls
REALUSER_DB_PATH=...     # Path to results.db (default: real_user_study.sqlite)
```

---

## 3. LLM User Study — Simulation (`llm_user_study/`)

Replaces human participants with LLM agents to simulate the full study at scale. Three student models (GPT-5, LLaMA 3 8B, Mistral Small 24B) × three prompting strategies × 55 personas × 5 runs.

### Two Simulation Modes

| Mode | Script | Description |
|---|---|---|
| **e2eAgent** | `run_simulation.py` | Full pipeline: pretest → 10 learning steps → survey |
| **AgentJudge** | `survey_only_simulation.py` | Survey only, given a real learner's trajectory |

### Prompting Strategies (e2eAgent)

| Strategy | `--prompt-type` | Description |
|---|---|---|
| **Calibrated** | `detailed` | Difficulty scale, mastery, education-specific distractor rules |
| **Task-Constrained** | `general` | Mastery + education level, instruction not to overperform |
| **Role-Anchored** | `simple` | Minimal: "You are a student. You are NOT an LLM." |

### Run Simulations

```bash
# e2eAgent — full simulation
python -m llm_user_study.run_simulation --prompt-type detailed --model llama-3-8b --runs 5
python -m llm_user_study.run_simulation --prompt-type general  --model mistral-24b-instruct --runs 5
python -m llm_user_study.run_simulation --prompt-type simple   --model gpt-5-2025-08-07 --runs 5

# AgentJudge — survey only
python -m llm_user_study.survey_only_simulation --batch --model llama-3-8b --runs 5
```

### Export & Analyze

```bash
python llm_user_study/scripts/export_results/export_csvs.py
python llm_user_study/scripts/bootstrapping.py
python llm_user_study/scripts/export_results/plot_trajectories.py --agent e2e --prompt-type simple
python llm_user_study/scripts/export_results/plot_survey_histograms.py --agent e2e --prompt-type simple
```

### Compute Simulation Cost

```bash
python -m llm_user_study.scripts.compute_simulation_cost --agent e2e
python -m llm_user_study.scripts.compute_simulation_cost --agent judge
```

Pricing is configured in `llm_user_study/config.py` under `MODEL_PRICING`:

| Model | Input $/1M | Output $/1M | Provider |
|---|---|---|---|
| `gpt-5-2025-08-07` | $2.50 | $20.00 | OpenAI direct |
| `llama-3-8b` | $0.036 | $0.038 | OpenRouter (weighted avg) |
| `mistral-24b-instruct` | $0.053 | $0.095 | OpenRouter (weighted avg) |

### Additional Database Tables (simulation only)

- `llm_student_calls` — per-call token usage for student simulation (`call_type`: `question` / `survey`)
- `llm_survey_predictions` — AgentJudge survey outputs
- `survey_test_retest` — test-retest reliability runs (AgentJudge)

### Results Structure

```
llm_user_study/results/
├── e2eAgent/
│   ├── results_as_csv/
│   │   ├── calibrated/{gpt,llama,mistral}/run{1-5}_survey.csv
│   │   ├── task-constrained/{gpt,llama,mistral}/run{1-5}_survey.csv
│   │   ├── role-anchored/{gpt,llama,mistral}/run{1-5}_survey.csv
│   │   └── cost/                    # cost_by_model_strategy.csv, etc.
│   ├── learning_trajectories/
│   └── survey_histograms/
├── JudgeAgent/
│   ├── results_as_csv/
│   │   ├── {gpt,llama,mistral}/run{1-5}_survey.csv
│   │   └── cost/
│   └── survey_histograms/
└── alignment_results/               # Bootstrap alignment & compatibility tables
```

---

## Dependencies

- Python 3.8+
- PyTorch
- OpenAI / OpenRouter API access (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`)
- Gymnasium
- NumPy, Pandas, Matplotlib, Seaborn
- LangChain, SQLAlchemy, Streamlit

Install with:

```bash
pip install -r Final_Requirements.txt
```
