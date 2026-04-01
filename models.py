from openenv.core.env_server import Action, Observation, State
from typing import Literal, Optional

TaskDifficulty = Literal["easy", "medium", "hard"]

class NumberGuessAction(Action):
    """Player's guess."""
    guess: int

class NumberGuessObservation(Observation):
    """Observation returned after each step."""
    feedback: str                     # "too high", "too low", "correct", "invalid"
    attempts_remaining: int
    message: str
    progress_score: float = 0.0       # 0.0-1.0 partial progress (for shaping)
    final_grade: Optional[float] = None  # 0.0-1.0 only on done=True (grader)

class NumberGuessState(State):
    """Internal state (hidden from agent)."""
    episode_id: str = ""
    step_count: int = 0
    max_attempts: int = 10
    secret_number: int = 0
    difficulty: TaskDifficulty = "medium"
