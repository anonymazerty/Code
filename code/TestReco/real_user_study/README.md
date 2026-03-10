# Real-User Study: Adaptive Question Recommendation System

## 1. Project Overview

This repository contains the full codebase for a **real-user study**
evaluating an **adaptive educational question recommender system**. The
goal of the study is to analyze how different recommendation strategies
impact learning experience, perceived difficulty, and user satisfaction.

The project is composed of **two main parts**:

1.  **Orchestrator System (backend logic)**
    -   Decides which recommendation strategy (policy) to apply at each
        learning step.
    -   Uses trained reinforcement-learning policies.
    -   Relies on an LLM (tool-call orchestrator) to reason
        over policy selection.
2.  **Web Application (Streamlit)**
    -   User-facing interface used by participants.
    -   Manages the full study flow: consent, qualification test,
        learning, survey, and completion.
    -   Uses the orchestrator and trained policies internally.

The **web application** needs to be deployed publicly
------------------------------------------------------------------------

## 2. High-Level User Flow

Each participant goes through the following phases:

1.  Landing page\
2.  Login / registration\
3.  Consent form\
4.  Qualification test (pretest)\
5.  Eligibility check and results\
6.  Adaptive learning phase\
7.  Post-study survey\
8.  Completion page with Prolific completion code

At any moment, the participant may end the session by clicking **End
session**.

------------------------------------------------------------------------

## 3. Repository Structure

    .
    ├── real_user_study/          # Streamlit web application (DEPLOY THIS)
    ├── orchestrator/             # Orchestrator logic (policy selection)
    ├── tools/                    # Policy factory and wrappers
    ├── results/                  # Trained policies (models + metadata)
    ├── configs/                  # JSON schemas for orchestrator tools
    └── Final_Requirements.txt    # REQUIRED: full Python environment used for deployment

Dependency management:

The application must be deployed using the exact Python environment
defined in Final_Requirements/.

This directory contains the complete list of packages installed in the
Python environment used during development and testing.

Deployments that do not install dependencies from
Final_Requirements/ may result in runtime errors or incompatibilities
(notably for the orchestrator and policy execution).
------------------------------------------------------------------------

## 4. Web Application (`real_user_study/`)

This folder contains the **Streamlit application** used by participants.

### Entry point

    real_user_study/app.py

Run locally with:

``` bash
python -m streamlit run real_user_study/app.py --server.runOnSave=true
```
after activating the pythonic env.

This file: - Defines all UI phases - Handles Streamlit session state -
Logs user actions and answers - Calls the orchestrator during the
learning phase - Displays the final Prolific completion code

------------------------------------------------------------------------

## 5. Orchestrator System

### Purpose

The orchestrator dynamically selects **which trained policy** should
recommend questions to the learner, based on the learner's current state
(mastery, failed questions, recent performance).

### Tool-Call Orchestrator

The study uses the **Tool-Call Orchestrator**:

    orchestrator/tool_call_orchestrator.py

Characteristics: - Uses an LLM to reason about policy selection - Can
simulate policy outcomes before choosing - Makes a final decision on
which policy to apply

The orchestrator is instantiated automatically by the web app during the
learning phase.


### Important note
Environment variables and API keys (REQUIRED):
The orchestrator relies on a private OpenRouter API key to function.
This key is provided via a .env file and must not be committed or
shared publicly.


------------------------------------------------------------------------

## 6. Trained Policies (`results/`)

The `results/` folder contains **pre-trained reinforcement-learning
policies**.

Each policy folder contains: - `config.json` - A trained model file
(`.pt` / `.pth`) - `policy_level_profile.json` (used by the orchestrator
for metadata)

These folders **must be present on the deployment server**.

------------------------------------------------------------------------

## 7. Survey and Likert Questions

After the learning phase, participants complete a survey with 7
Likert-scale questions (1--5), covering: - Accomplishment - Effort -
Mental demand - Perceived controllability - Temporal demand -
Frustration - Trust

The survey UI is implemented in:

    real_user_study/ui_components.py

------------------------------------------------------------------------

## 8. Database and Logging

-   Uses SQLite by default
-   Stores:
    -   User sessions
    -   Question attempts
    -   Learning steps
    -   Survey responses
-   Suitable for later statistical analysis

Ensure write permissions on the deployment machine.

------------------------------------------------------------------------

## 9. Prolific Completion Codes

### Current implementation

Completion codes are generated in:

    completion_codes.py

Current behavior: - Each user is deterministically assigned a **unique
completion code** - The code is shown **only after successful survey
submission** - Prevents duplicate payment claims

### Important note

Handling of **early session termination** (before survey completion) is
not done. If partial payment or alternative logic is
required, it should be implemented inside `completion_codes.py`.

No other part of the system needs to be modified for Prolific payment
logic.


