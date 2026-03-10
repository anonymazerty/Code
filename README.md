# CROWDvsLLM: Can LLM Agents Replace Crowd Workers in Educational User Studies?

This repository contains the code, prompts, and results for **CROWDvsLLM**, a study investigating whether LLM-based agents can serve as reliable proxies for human participants in educational system evaluations. The project spans two complementary use cases: **adaptive test recommendation** (TestReco) and **adaptive quiz composition** (QuizComp), each evaluated through both a real crowd-sourced user study and large-scale LLM simulations.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Use Case 1: TestReco — Adaptive Question Recommendation](#use-case-1-testreco--adaptive-question-recommendation)
4. [Use Case 2: QuizComp — Adaptive Quiz Composition](#use-case-2-quizcomp--adaptive-quiz-composition)
5. [LLM Agents](#llm-agents)
6. [Prompting Strategies](#prompting-strategies)
7. [Evaluation Framework](#evaluation-framework)
8. [Results](#results)
9. [Prompts](#prompts)
10. [Getting Started](#getting-started)

---

## Overview

Educational systems increasingly rely on user studies to validate their effectiveness, but recruiting and managing human participants is expensive, slow, and hard to scale. This project asks: **can LLM agents faithfully replicate human behavior in educational settings?**

We address this through two distinct use cases, each involving:

- A **real user study** with crowd workers recruited via Prolific
- An **LLM simulation study** where LLM agents replay the same tasks using persona-calibrated prompts
- A **comparative analysis** measuring behavioral alignment between humans and agents

### Key Concepts

| Concept | Description |
|---|---|
| **e2eAgent** | End-to-end agent: an LLM agent that completes the full study pipeline (interactions + survey) |
| **JudgeAgent** | Survey-only agent: given a real participant's trajectory, the LLM predicts what survey responses the participant would give |

---

## Repository Structure

```
CROWDvsLLM/
├── README.md                         # This file
│
├── code/
│   ├── TestReco/                     # Adaptive question recommendation system
│   │   ├── agents/                   # RL agents (SARSA, A2C, PPO)
│   │   ├── envs/                     # Educational environment (Gymnasium)
│   │   ├── orchestrator/             # LLM-based orchestrator (3 variants)
│   │   ├── real_user_study/          # Streamlit web app for human study
│   │   ├── llm_user_study/           # LLM simulation framework
│   │   ├── results/                  # Trained RL policy checkpoints
│   │   ├── train_evaluate_policy.py  # RL policy training
│   │   ├── main.py                   # Standalone orchestrator evaluation
│   │   └── viz.py                    # Visualization and analysis
│   │
│   └── QuizComp/                     # Adaptive quiz composition system
│       ├── app/                      # FastAPI backend + RL agents
│       ├── quizcomp_ui/              # Streamlit user study interface
│       ├── quizcomp_llm_study/       # LLM simulation framework
│       ├── data/                     # MCQ dataset
│       └── models/                   # Trained RL checkpoints
│
├── prompts/
│   ├── TestReco/                     # Prompting strategy templates (TestReco)
│   └── QuizComp/                     # Prompting strategy templates (QuizComp)
│
└── results/
    ├── TestReco/                     # TestReco simulation results
    │   ├── e2eAgent/                 # Full pipeline simulation outputs
    │   ├── JudgeAgent/               # Survey-only simulation outputs
    │   ├── alignment_results/        # Alignment and statistical tests
    │   └── db/                       # Source databases
    │
    └── QuizComp/                     # QuizComp simulation results
        ├── e2eAgent/                 # Full pipeline simulation outputs
        ├── JudgeAgent/               # Survey-only simulation outputs
        └── dbs/                      # Source databases
```

---

## Use Case 1: "TestReco" Adaptive Question Recommendation

TestReco is a **multi-objective adaptive question recommendation** system. An RL-trained orchestrator recommends sequences of math questions to learners, balancing three objectives: **performance** (accuracy), **gap** (remediation of weak areas), and **aptitude** (stretching potential).

### Real User Study Flow

1. Landing → Login → Consent
2. **Pretest**: 12 qualification questions from a math benchmark
3. Eligibility check 
4. **Adaptive learning**: 10 steps × 5 questions, orchestrator-driven
5. **Post-study survey**: NASA-TLX style (Likert items, 1–5)
6. Completion code for Prolific to get paid

### LLM Simulation

The simulation replaces human learners with LLM agents that answer questions and complete surveys using personas prompts. Three student models are used across three prompting strategies and 55 personas with 5 runs each.

For details, see [code/TestReco/README.md](code/TestReco/README.md).

---

## Use Case 2: "QuizComp" Adaptive Quiz Composition

QuizComp is an **adaptive quiz composition** system. Teachers specify topic and difficulty preferences; an RL agent searches a universe of candidate quizzes and returns the best match. Teachers iteratively refine their preferences until satisfied.

### Teacher Interaction Loop

1. Set topic and difficulty distribution preferences
2. System generates a quiz and reports match quality
3. Teacher decides: **Accept**, **Improve** (retry same params), or **Fresh** (change params)
4. Repeat until satisfied
5. **Post-study survey**

### LLM Simulation

LLM agents simulate 65 teacher personas extracted from the real user study. Each persona replays the full composition task and provides survey responses.

For details, see [code/QuizComp/README.md](code/QuizComp/README.md).

---

### Participants Models (TestReco & QuizComp)

| Model | Provider | Input $/1M | Output $/1M |
|---|---|---|---|
| GPT-5 | OpenAI (direct) | $2.50 | $20.00 |
| LLaMA 3 8B | OpenRouter | $0.036 | $0.038 |
| Mistral Small 24B | OpenRouter | $0.053 | $0.095 |

Each model is evaluated across all prompting strategies, all personas, and 5 independent runs to assess consistency and reliability.

---

## Prompting Strategies

Three prompting strategies vary in how much behavioral guidance they provide to the LLM agent:

### 1. Calibrated

The most detailed strategy. Provides explicit behavioral rules calibrated to the agent's education level:
- Difficulty-aware answer generation (mastery-based error modeling)
- Education-specific distractor selection patterns
- Detailed instructions on how to simulate mistakes realistically

### 2. Task-Constrained

Moderate detail. Provides the task structure and key constraints:
- Mastery level and education level context
- Instruction not to overperform relative to the persona's profile
- No explicit distractor rules

### 3. Role-Anchored

Minimal guidance. Anchors the agent in a role identity:
- "You are a student. You are NOT an LLM."
- Education level and mastery as context
- No behavioral rules: relies on the model's implicit role-playing

All prompt templates are available in the [prompts/](prompts/) directory.

---

## Evaluation Framework

### Alignment Metrics

The study evaluates behavioral alignment between human and LLM distributions using:

- **Wilcoxon Rank-Sum Tests** 
- **Bootstrap Alignment Scores** 
- **Kruskal-Wallis Tests** 
- **Group Medians**

### What Is Measured

| Dimension | Metric | Source |
|---|---|---|
| **Task behavior** | Learning trajectories, answer accuracy, mastery progression | e2eAgent |
| **Subjective perception** | NASA-TLX survey items (accomplishment, effort, mental demand, controllability, temporal demand, frustration, trust) | e2eAgent + JudgeAgent |
| **Consistency** | Test-retest reliability across 5 runs per persona | JudgeAgent |
| **Cost efficiency** | Token usage and API cost per simulation | Both agents |

---

## Results

Pre-computed results for both use cases are provided in the [results/](results/) directory:

### TestReco Results

```
results/TestReco/
├── e2eAgent/
│   ├── results_as_csv/           # Per-model, per-strategy survey CSVs
│   ├── learning_trajectories/    # Mastery progression plots
│   └── survey_histograms/        # Survey response distributions
├── JudgeAgent/
│   ├── results_as_csv/           # Survey prediction CSVs
│   ├── learning_trajectories/    # Trajectory comparisons
│   ├── survey_histograms/        # Distribution comparison plots
│   └── stats/                    # Statistical test outputs
├── alignment_results/
│   ├── alignment_results.csv     # Bootstrap alignment scores
│   └── wilcoxon_results.csv      # Per-item statistical tests
└── db/                           # Source SQLite databases
```

### QuizComp Results

```
results/QuizComp/
├── e2eAgent/
│   ├── Results_as_csvs/          # Per-strategy survey CSVs
│   ├── survey_histograms/        # Distribution plots
│   ├── Calibrated/               # Calibrated strategy outputs
│   ├── Role-Anchored/            # Role-Anchored strategy outputs
│   └── Task-Constrained/         # Task-Constrained strategy outputs
├── JudgeAgent/
│   ├── Results_as_csvs/          # Survey prediction CSVs
│   ├── survey_histograms/        # Distribution plots
│   └── statistical_tests_report.txt
├── dbs/                          # Source SQLite databases
└── group_medians_summary.csv     # Aggregated median comparisons
```

---

## Prompts

The exact prompting templates used for each strategy and use case are provided for full reproducibility:

```
prompts/
├── TestReco/
│   ├── calibrated.py             # Calibrated strategy (detailed behavioral rules)
│   ├── task-constrained.py       # Task-Constrained strategy (moderate guidance)
│   ├── role-anchored.py          # Role-Anchored strategy (minimal guidance)
│   └── survey.py                 # Survey completion prompt
│
└── QuizComp/
    ├── calibrated_prompt.py      # Calibrated strategy for quiz composition
    ├── task_constrained.py       # Task-Constrained strategy
    ├── role_anchored.py          # Role-Anchored strategy
    └── survey_prompt.py          # Survey completion prompt
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- API keys: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`

### TestReco

```bash
cd code/TestReco
pip install -r Final_Requirements.txt

# Train RL policies
python3 train_evaluate_policy.py --agent ppo --objectives "performance" "aptitude" "gap" --train_episodes 10000
python3 train_evaluate_policy.py --agent ppo --objectives "performance"  --train_episodes 10000
python3 train_evaluate_policy.py --agent ppo --objectives "aptitude"  --train_episodes 10000
python3 train_evaluate_policy.py --agent ppo --objectives "gap" --train_episodes 10000

# Run the real user study (Streamlit)
python -m streamlit run real_user_study/app.py

# Run LLM simulation (e2eAgent)
python -m llm_user_study.run_simulation --prompt-type detailed --model llama-3-8b --runs 5

# Run LLM simulation (JudgeAgent)
python -m llm_user_study.survey_only_simulation --batch --model llama-3-8b --runs 5
```

### QuizComp

```bash
cd code/QuizComp
pip install -r requirements.txt

# Start the backend
cd app && uvicorn main:app --port 8000

# Run the user study UI (in another terminal)
cd quizcomp_ui && streamlit run str_app.py

# Run LLM simulation
cd quizcomp_llm_study && python run_interactive_simulation.py
```

For full documentation of each component, refer to the README files in [code/TestReco/](code/TestReco/README.md) and [code/QuizComp/](code/QuizComp/README.md).
