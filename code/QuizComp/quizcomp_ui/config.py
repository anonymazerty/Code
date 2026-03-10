# quizcomp_ui/config.py
import pandas as pd

API_BASE_DEFAULT = "http://127.0.0.1:8000"


# IMPORTANT: this order should be kept consistent everywhere (UI vector order).
TOPICS = [
    (4,  "Differentiation", 153),
    (5,  "Integration", 310),
    (7,  "Fundamental Mathematics", 190),
    (10, "Complex Numbers", 84),
    (12, "Optimization", 116),
    (16, "Differential Equations", 52),
    (17, "Probability", 70),
    (18, "Linear Algebra", 196),
    (26, "Numerical Methods", 41),
    (28, "Discrete Mathematics", 116),
]

NUM_DIFFICULTIES = 6

# Hidden defaults
DEFAULT_MCQS = ["data/math.csv"]
DEFAULT_NUM_QUIZZES = 10000
DEFAULT_NUM_TOPICS = 10
DEFAULT_LIST_TOPICS = []
DEFAULT_TOPIC_MODE = 1
DEFAULT_LEVEL_MODE = 1
DEFAULT_ORDER_LEVEL = 2

DEFAULT_MODEL_PATH = "models/dqn_t4_math_r2"
DEFAULT_ALFA = 0.5

def detect_difficulties(csv_path="data/math.csv"):
    df = pd.read_csv(csv_path, sep=";")
    col = "difficulty_level" if "difficulty_level" in df.columns else "Level"
    vals = sorted(df[col].dropna().astype(int).unique().tolist())
    return vals