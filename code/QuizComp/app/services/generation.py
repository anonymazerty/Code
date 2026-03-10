# app/services/generation.py
import os
import json
import numpy as np
import pandas as pd
from app.schemas.generation import QuizGenerationRequest
from uuid import UUID

# We intentionally DO NOT filter by teacher-selected topics in generation.
# The universe is generated from the dataset topic pool (or from numTopics random topics when listTopics is empty).


def generate_quiz_universe(req: QuizGenerationRequest, uuid: UUID) -> str:
    """
    Generate a universe of quizzes.

    IMPORTANT (desired behavior):
      - Do NOT filter by req.listTopics (we ignore it).
      - Universe should be stable wrt topics/difficulties and only regenerated when numMCQs changes.

    Returns:
      Path to universe json file: data/{uuid}/universe_{uuid}.json
    """
    if not os.path.exists("data"):
        os.makedirs("data")
    os.makedirs(f"data/{uuid}", exist_ok=True)

    # Write request to yml (for debugging/repro)
    with open(f"data/{uuid}/request_{uuid}.yml", "w") as f:
        for key, value in req.dict().items():
            f.write(f"{key}: {value}\n")

    try:
        # IGNORE req.listTopics on purpose
        mcqs_list, num_topics = load_mcqs(req.MCQs, req.numTopics, listTopics=[], uuid=uuid)
    except Exception as e:
        raise ValueError(f"Error loading MCQs: {e}")

    universe, topic_to_idx = sample_combinations(
        mcqs_list,
        req.numQuizzes,
        req.numMCQs,
        num_topics,
        req.topicMode,
        req.levelMode,
        req.orderLevel,
    )
    output_path = create_quiz_dataframe(universe, num_topics, topic_to_idx, req.numMCQs, uuid)

    return output_path


def load_mcqs(urls: list[str], numTopics: int, listTopics: list[str], uuid: UUID) -> tuple[list[dict], int]:
    """
    Load MCQs from the provided URLs and parse them into a list of dictionaries.

    We keep listTopics argument for compatibility, but generation passes listTopics=[].
    """
    if not urls:
        raise ValueError("No URLs provided for loading MCQs.")

    dataframes = []
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

    for path in urls:
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(path, sep=";", encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"Could not read file {path} with any of the tried encodings: {encodings}")
        dataframes.append(df)

    mcqs_df = pd.concat(dataframes, ignore_index=True)
    if mcqs_df.empty:
        raise ValueError("No data found in the provided dataset.")

    try:
        mcqs_list, num_topics = parse_dataset(mcqs_df, numTopics, listTopics, uuid)
    except ValueError as e:
        print(f"Error parsing dataset: {e}")
        mcqs_list = []
        num_topics = 0

    print(f"Generated {len(mcqs_list)} unique MCQs from {len(urls)} files with {num_topics} topics.")
    return mcqs_list, num_topics


def parse_dataset(df: pd.DataFrame, numTopics: int, listTopics: list[str], uuid: UUID) -> tuple[list[dict], int]:
    """
    Parse the dataset into a list of MCQs with fields: id, topic, difficulty.

    If listTopics is empty: choose numTopics random topics from dataset.
    (In your desired behavior, listTopics will always be empty from generation.)
    """
    df.rename(columns={"topic_id": "topic", "Level": "difficulty"}, inplace=True)

    df.rename(columns={"id": "mcq_id"}, inplace=True)
    df.rename(columns={"difficulty_level": "difficulty"}, inplace=True)
    df["option_a"] = df["correct_answer"]
    df.rename(
        columns={
            "correct_answer": "correct_option",
            "answer2": "option_b",
            "answer3": "option_c",
            "answer4": "option_d",
        },
        inplace=True,
    )
    df["topic"] = df["topic"].astype(int)

    df["id"] = df.index
    df["difficulty"] = df["difficulty"]

    # mapping between topic id and topic name (only needed when listTopics is used)
    topic_mapping = (
        df[["topic", "topic_name"]]
        .drop_duplicates()
        .set_index("topic")["topic_name"]
        .to_dict()
    )

    unique_topics = df["topic"].unique()
    unique_topic_names = df["topic_name"].unique()

    if listTopics:
        selected_topic_names = [
            topic for topic in unique_topic_names
            if str(topic).lower() in [t.lower() for t in listTopics]
        ]
        selected_topics = [
            topic for topic in unique_topics
            if topic_mapping.get(topic, "").lower() in [t.lower() for t in listTopics]
        ]
        num_topics = len(selected_topics)
        if not selected_topics:
            raise ValueError("No valid topics found in the provided list.")
    else:
        num_topics = numTopics
        if num_topics <= 0:
            raise ValueError("Number of topics must be greater than 0.")
        if num_topics > len(unique_topics):
            print(
                f"Warning: Requested {num_topics} topics but only {len(unique_topics)} available. Using all available topics."
            )
            num_topics = len(unique_topics)
        selected_topics = np.random.choice(unique_topics, num_topics, replace=False)
        print(f"Selected topics: {selected_topics}")

    mcqs = df[df["topic"].isin(selected_topics)]

    mcqs.to_csv(f"data/{uuid}/mcqs_{uuid}.csv", index=False)
    print(f"Generated {len(mcqs)} unique MCQs")

    mcqs_list = []
    for _, row in mcqs.iterrows():
        mcq_dict = {
            "id": row["id"],
            "topic": row["topic"],
            "difficulty": row["difficulty"],
        }
        mcqs_list.append(mcq_dict)

    return mcqs_list, num_topics


def order_level_quizzes(universe: list[list[dict]], orderLevel: int) -> list[list[dict]]:
    if orderLevel == 0:
        return sorted(universe, key=lambda quiz: sum(mcq["difficulty"] for mcq in quiz))
    elif orderLevel == 1:
        return sorted(universe, key=lambda quiz: sum(mcq["difficulty"] for mcq in quiz), reverse=True)
    else:
        return universe


def sample_combinations(
    mcq_list: list[dict],
    numQuizzes: int,
    numMCQs: int,
    num_topics: int,
    topicMode: bool,
    levelMode: bool,
    orderLevel: int,
) -> tuple[list[list[dict]], dict]:
    data_dict = initialize_generator(mcq_list, numMCQs, num_topics, num_difficulties=6)
    print(
        f"Initialized generator with the following parameters:\n"
        f"  - Number of topics: {data_dict['num_topics']}\n"
        f"  - Number of difficulties: {data_dict['num_difficulties']}\n"
        f"  - Quiz size: {data_dict['quiz_size']}\n"
        f"  - Topics to index mapping: {data_dict['topic_to_idx']}\n"
        f"  - Difficulty to index mapping: {data_dict['difficulty_to_idx']}"
    )

    universe = []
    seen_quizzes = set()

    max_attempts = 1000
    attempts = 0

    while len(universe) < numQuizzes and attempts < max_attempts:
        quiz = generate_quiz(data_dict, topicMode, levelMode)
        quiz_key = tuple(sorted(mcq["id"] for mcq in quiz))

        if quiz_key not in seen_quizzes:
            universe.append(quiz)
            seen_quizzes.add(quiz_key)
            attempts = 0
        else:
            attempts += 1

    if len(universe) < numQuizzes:
        print(f"Warning: Could only generate {len(universe)} unique quizzes out of requested {numQuizzes}")

    universe = order_level_quizzes(universe, orderLevel)
    return universe, data_dict["topic_to_idx"]


def initialize_generator(mcq_list: list[dict], quiz_size: int, num_topics: int, num_difficulties: int = 6) -> dict:
    mcqs = mcq_list

    topic_to_idx = {topic: i for i, topic in enumerate(sorted(set(m["topic"] for m in mcqs)))}
    difficulty_to_idx = {i: i for i in range(num_difficulties)}

    mcqs_by_topic_diff = {}
    for mcq in mcqs:
        topic = mcq["topic"]
        difficulty = int(mcq["difficulty"]) - 1
        if topic not in mcqs_by_topic_diff:
            mcqs_by_topic_diff[topic] = {}
        if difficulty not in mcqs_by_topic_diff[topic]:
            mcqs_by_topic_diff[topic][difficulty] = []
        mcqs_by_topic_diff[topic][difficulty].append(mcq)

    return {
        "mcqs_by_topic_diff": mcqs_by_topic_diff,
        "quiz_size": quiz_size,
        "num_topics": num_topics,
        "num_difficulties": num_difficulties,
        "topic_to_idx": topic_to_idx,
        "difficulty_to_idx": difficulty_to_idx,
    }


def generate_quiz(data_dict: dict, topicMode: bool, levelMode: bool) -> list[dict]:
    quiz = []
    mcqs_by_topic_diff = data_dict["mcqs_by_topic_diff"]
    quiz_size = data_dict["quiz_size"]
    num_topics = data_dict["num_topics"]

    topics = list(mcqs_by_topic_diff.keys())
    max_attempts = 100
    quiz_built = False
    counter = 0
    max_counter = 100

    while not quiz_built and counter < max_counter:
        attempts = 0
        chosen_difficulty = None

        if topicMode:
            sampled_topics = np.random.choice(topics, size=num_topics, replace=False)
        else:
            sampled_topics = [np.random.choice(topics)]

        quiz.clear()

        while len(quiz) < quiz_size and attempts < max_attempts:
            topic = np.random.choice(sampled_topics)

            if levelMode:
                difficulties = list(mcqs_by_topic_diff[topic].keys())
                if not difficulties:
                    attempts += 1
                    continue
                difficulty = np.random.choice(difficulties)
            else:
                if chosen_difficulty is None:
                    available_difficulties = set(mcqs_by_topic_diff[sampled_topics[0]].keys())
                    for t in sampled_topics[1:]:
                        available_difficulties &= set(mcqs_by_topic_diff[t].keys())
                    if not available_difficulties:
                        attempts += 1
                        continue
                    chosen_difficulty = np.random.choice(list(available_difficulties))
                difficulty = chosen_difficulty

            mcqs_pool = mcqs_by_topic_diff[topic].get(difficulty, [])
            if not mcqs_pool:
                attempts += 1
                continue
            mcq = np.random.choice(mcqs_pool)
            if mcq in quiz:
                attempts += 1
                continue

            quiz.append(mcq)
            attempts = 0

        if len(quiz) == quiz_size:
            quiz_built = True
        else:
            continue

        counter += 1

    if len(quiz) < quiz_size:
        raise ValueError(
            f"Could not generate a quiz of size {quiz_size} after multiple attempts. Not enough MCQs available."
        )
    return quiz


def create_quiz_dataframe(universe: list[str], num_topics: int, topic_to_idx: dict, numMCQs: int, uuid: UUID) -> str:
    quiz_data = []
    universe_array = []
    num_difficulties = 6

    for quiz_idx, quiz in enumerate(universe):
        topic_dist = np.zeros(num_topics)
        difficulty_dist = np.zeros(num_difficulties)

        mcq_ids = [mcq["id"] for mcq in quiz]
        for mcq in quiz:
            topic = mcq["topic"]
            difficulty = mcq["difficulty"] - 1
            topic_dist[topic_to_idx[topic]] += 1
            difficulty_dist[difficulty] += 1

        topic_dist = topic_dist / len(quiz)
        difficulty_dist = difficulty_dist / len(quiz)

        row_data = {
            "quiz_id": quiz_idx,
            **{f"mcq_{i+1}": mcq_id for i, mcq_id in enumerate(mcq_ids)},
            **{f"topic_coverage_{i}": cov for i, cov in enumerate(topic_dist)},
            **{f"difficulty_coverage_{i}": cov for i, cov in enumerate(difficulty_dist)},
        }
        quiz_data.append(row_data)

        combined_dist = np.concatenate([topic_dist, difficulty_dist])
        universe_array.append(combined_dist)

    quizzes_df = pd.DataFrame(quiz_data)
    quizzes_df.to_csv(f"data/{uuid}/quizzes_{uuid}.csv", index=False)

    universe_array = np.array(universe_array)
    output_path = f"data/{uuid}/universe_{uuid}.json"
    with open(output_path, "w") as f:
        json.dump(universe_array.tolist(), f)

    print(f"Generated universe with {len(universe_array)} quizzes, each containing {numMCQs} MCQs.")
    return output_path
