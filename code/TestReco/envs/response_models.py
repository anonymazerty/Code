"""
Response models for simulating student responses.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from EduCDM import GDIRT


def transform(x, y, z, batch_size, **params):
    """Transform data into PyTorch DataLoader format for GDIRT."""
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


class BaseResponseModel(ABC):
    """Base class for all response models."""

    def __init__(self, num_skills: int, seed: Optional[int] = None):
        """Initialize base response model with local random generator."""
        self.num_skills = num_skills
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def predict_response(self, *args, **kwargs) -> Tuple[bool, float]:
        """Predict student response."""
        pass

    def reset(self):
        """Reset the model's internal state."""
        pass

    def set_seed(self, seed: int):
        """Set new seed for the random generator."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)


class BKTModel(BaseResponseModel):
    """Bayesian Knowledge Tracing model for simulating student responses."""

    def __init__(self, num_skills: int, seed: Optional[int] = None):
        super().__init__(num_skills, seed)
        
        # BKT parameters for each skill
        self.p_init = np.full(num_skills, 0.3)  # P(L₀): Initial probability of mastery
        self.p_transit = np.full(num_skills, 0.1)  # P(T): Probability of learning
        self.p_slip = np.full(num_skills, 0.1)  # P(S): Probability of slip
        self.p_guess = np.full(num_skills, 0.2)  # P(G): Probability of guess

        # Current belief state
        self.p_mastered = self.p_init.copy()  # P(L): Current probability of mastery

    def update_belief(self, skills_tested: np.ndarray, correct: bool):
        """Update belief state based on observed response."""
        if len(skills_tested) == 0:
            return

        for skill_idx in skills_tested:
            p_learned = self.p_mastered[skill_idx]

            if correct:
                # P(L|correct) = P(L)*(1-P(S)) / (P(L)*(1-P(S)) + (1-P(L))*P(G))
                p_learned = (p_learned * (1 - self.p_slip[skill_idx])) / (
                    p_learned * (1 - self.p_slip[skill_idx])
                    + (1 - p_learned) * self.p_guess[skill_idx]
                )
            else:
                # P(L|incorrect) = P(L)*P(S) / (P(L)*P(S) + (1-P(L))*(1-P(G)))
                p_learned = (p_learned * self.p_slip[skill_idx]) / (
                    p_learned * self.p_slip[skill_idx]
                    + (1 - p_learned) * (1 - self.p_guess[skill_idx])
                )

            # Apply learning probability
            p_learned = p_learned + (1 - p_learned) * self.p_transit[skill_idx]

            self.p_mastered[skill_idx] = p_learned

    def predict_response(self, skills_tested: np.ndarray) -> Tuple[bool, float]:
        """Predict whether a student will answer correctly."""
        if len(skills_tested) == 0:
            logging.info(f"No skills tested, returning random student response.")
            random_number = self.rng.random()
            return bool(random_number > 0.5), 0.5

        # Calculate average mastery probability for tested skills
        p_mastered_avg = np.mean(self.p_mastered[skills_tested])

        # P(correct) = P(L)*(1-P(S)) + (1-P(L))*P(G)
        p_slip_avg = np.mean(self.p_slip[skills_tested])
        p_guess_avg = np.mean(self.p_guess[skills_tested])

        p_correct = (
            p_mastered_avg * (1 - p_slip_avg) + (1 - p_mastered_avg) * p_guess_avg
        )
        random_number = self.rng.random()
        print(f"Random number: {random_number}, p_correct: {p_correct}")

        return bool(random_number < p_correct), p_correct

    def reset(self, mastery: np.ndarray):
        """Reset the model's internal state."""
        # Reset mastery probabilities to initial state
        self.p_mastered = mastery.copy()


class GDIRTModel(BaseResponseModel):
    """GDIRT (Generalized Deterministic Input, Noisy "And" gate Item Response Theory) model using EduCDM."""

    def __init__(self, num_skills: int, num_questions: int, seed: Optional[int] = None):
        super().__init__(num_skills, seed)
        
        self.num_questions = num_questions
        self.num_skills = num_skills
        
        # Initialize GDIRT model
        self.cdm = GDIRT(2, num_questions)  # 2 for 2PL model, num_questions for number of items
        
        # Training data storage: list of (student_id, question_id, correct)
        self.training_data = []
        
        # Flag to track if model has been trained
        self.is_trained = False

    def add_training_data(self, difficulty: float, correct: bool, student_id: int = 0):
        """Add new training data point."""
        self.training_data.append((student_id, difficulty, int(correct)))

    def train_model(self, epochs: int = 10):
        """Train the GDIRT model with current training data."""
        if len(self.training_data) < 5:
            logging.warning("Not enough training data for GDIRT training")
            return
        
        try:
            # Convert training data to the format expected by GDIRT
            # training_data format: [(student_id, question_id, correct), ...]
            student_ids = [item[0] for item in self.training_data]
            difficulties = [item[1] for item in self.training_data]
            corrects = [item[2] for item in self.training_data]
            
            # Create DataLoader using transform function
            batch_size = min(32, len(self.training_data))  # Use smaller batch size
            train_loader = transform(student_ids, difficulties, corrects, batch_size)
            
            # Train the model
            self.cdm.train(train_loader, epoch=epochs)
            self.is_trained = True
            
            logging.info(f"GDIRT model trained with {len(self.training_data)} data points")
        except Exception as e:
            logging.error(f"Error training GDIRT model: {e}")
            self.is_trained = False

    def predict_response(self, difficulties: List[float], student_id: int = 0) -> List[Tuple[bool, float]]:
        """
        Predict responses for a batch of questions using GDIRT model.
        
        Args:
            difficulties: List of difficulties to predict
            
        Returns:
            List[Tuple[bool, float]]: List of (correct, probability) for each question
        """
        if not self.is_trained:
            raise ValueError("GDIRT model is not trained")
        
        # Create test data for batch prediction
        student_ids = [student_id] * len(difficulties)  # All same student
        difficulties = difficulties
        corrects = [0] * len(difficulties)  # Dummy values
        
        # Create DataLoader
        batch_size = min(32, len(difficulties)) 
        test_loader = transform(student_ids, difficulties, corrects, batch_size)
        
        # Use eval for batch prediction
        preds = self.cdm.eval_prediction(test_loader)
        
        # Extract probabilities and generate responses
        results = []
        for i in range(len(difficulties)):
            # Get probability from the prediction
            prob = float(preds[i])
            # Use threshold-based decision: correct if prob > 0.9, otherwise incorrect
            correct = 1 if prob > 0.9 else 0
            
            results.append((bool(correct), prob))
        
        return results
            
    

    def reset(self, mastery: np.ndarray):
        """Reset the model's internal state."""
        # Reset training data
        self.training_data = []
        self.is_trained = False




class IRTModel(BaseResponseModel):
    """2PL Item Response Theory model for simulating student responses (no guessing parameter)."""

    def __init__(self, num_skills: int, temperature: float = 1.0, seed: Optional[int] = None):
        super().__init__(num_skills, seed)
        
        # Initialize student ability parameters (theta) for each skill
        self.abilities = np.zeros(num_skills)  # θ ~ N(0, 1)
        self.temperature = temperature

        # Question parameters (3PL model)
        self.difficulties = {}  # b (difficulty)
        self.discriminations = {}  # a (discrimination)
        # self.guessing_parameters = {}  # c (guessing parameter)

    def set_question_params_batch(
        self,
        difficulties: Dict[int, float],
        discriminations: Optional[Dict[int, float]] = None,
        guessing_parameters: Optional[Dict[int, float]] = None,
    ):
        """Set 2PL IRT parameters for multiple questions at once.

        Notes:
        - `difficulties` keys may be dataset question_ids (e.g., 102, 499) or dense indices (0..N-1),
          but they must match what is passed into `predict_response`.
        - Keys are normalized to `int` to avoid "102" vs 102 mismatches.
        """
        # Normalize / load difficulties
        for k, v in difficulties.items():
            self.difficulties[int(k)] = float(v)

        # Normalize / load discriminations (or fill defaults)
        if discriminations is not None:
            for k, v in discriminations.items():
                self.discriminations[int(k)] = float(v)

        # Fill any missing discriminations with a sane default
        for k in list(self.difficulties.keys()):
            if k not in self.discriminations:
                self.discriminations[k] = 4.0

        _ = guessing_parameters  # unused in 2PL

    def update_abilities(
        self, skills_tested: np.ndarray, correct: bool, learning_rate: float = 0.2
    ):
        """Update student abilities based on response."""
        if len(skills_tested) == 0:
            return

        update = learning_rate * (1.0 if correct else -1.0)
        self.abilities[skills_tested] += update
        # logging.info(f"Updated abilities: {self.abilities}")

    def predict_response(
        self, question_idx: int, skills_tested: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Predict whether a student will answer correctly using 2PL IRT model.
        Adapted for [0,1] range: P(θ) = 1/(1 + e^(-a(θ-b))).

        Robustness:
        - If `question_idx` is missing from configured parameters (e.g., CSV missing an id or id/index mismatch),
          we fall back to a default difficulty instead of raising KeyError.
        """
        if len(skills_tested) == 0:
            random_number = self.rng.random()
            return bool(random_number > 0.5), 0.5

        qk = int(question_idx)

        # Get question parameters; if missing, fall back safely (no crash)
        if qk in self.difficulties:
            b = float(self.difficulties[qk])
        else:
            if len(self.difficulties) > 0:
                b = float(np.median(np.array(list(self.difficulties.values()), dtype=float)))
            else:
                b = 0.5
            logging.warning(
                f"IRTModel: difficulty missing for question_idx={question_idx}. "
                f"Using fallback b={b:.4f}. This usually means the CSV lacks this id "
                f"or there is still a qid/index mismatch."
            )

        a = float(self.discriminations.get(qk, 4.0))

        theta = float(np.mean(self.abilities[skills_tested]))

        z = a * (theta - b + 0.4)
        p = float(1.0 / (1.0 + np.exp(-z)))

        random_number = self.rng.random()
        return bool(random_number < p), p

    def reset(self, mastery: np.ndarray):
        """Reset the model's internal state."""
        # Reset abilities using inverse sigmoid (logit) transformation
        # self.abilities = -np.log(1 / mastery - 1) / self.temperature
        # self.abilities = -3 + 6 * mastery
        self.abilities = mastery