# QuizComp UI - Study Interface

This is the user interface for the QuizComp study, built with Streamlit.

## Requirements

- Python 3.9 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository** (or download the files)

2. **Navigate to the quizcomp_ui directory**
   ```bash
   cd quizcomp_ui
   ```

3. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run database migration** (one-time setup)
   ```bash
   python migrate_add_q7.py
   ```

## Configuration

The application uses environment variables for configuration. You can set these in your environment or create a `.env` file:

- `QUIZCOMP_API_BASE` - Base URL for the QuizComp backend API (default: from config.py)
- `QUIZCOMP_DB_PATH` - Path to SQLite database file (default: `quizcomp_study.sqlite`)
- `QUIZCOMP_DB_URL` - Full database URL (optional, overrides DB_PATH)

## Running the Application

```bash
streamlit run str_app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

- `str_app.py` - Main Streamlit application
- `db.py` - Database models and setup
- `auth.py` - User authentication
- `study_ui.py` - Study session management
- `ui_components.py` - Reusable UI components for rendering quizzes
- `config.py` - Configuration constants
- `completion_codes_allocator.py` - Manages completion codes for study participants
- `migrate_add_q7.py` - Database migration script
- `requirements.txt` - Python package dependencies

## Database

The application uses SQLite by default. The database file (`quizcomp_study.sqlite`) will be created automatically on first run.

### Tables:
- `users` - User accounts
- `study_sessions` - Study session tracking
- `event_logs` - User interaction logs
- `compose_attempts` - Quiz composition attempts
- `survey_responses` - Post-study survey responses

## Features

- User authentication (login/register)
- Quiz parameter selection (topics, difficulty levels, number of questions)
- Interactive quiz composition with RL agent
- Quiz comparison (old vs new)
- Post-study survey
- Prolific integration for participant tracking

