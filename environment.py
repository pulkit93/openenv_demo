import random
import uuid
from typing import Optional
from ..models import NumberGuessAction, NumberGuessObservation, NumberGuessState, TaskDifficulty
from openenv.core.env_server import Environment

class NumberGuessingEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    TASK_CONFIG = {
        "easy":   {"min": 1, "max": 50,  "max_attempts": 15},
        "medium": {"min": 1, "max": 100, "max_attempts": 10},
        "hard":   {"min": 1, "max": 200, "max_attempts": 7},
    }

    def __init__(self):
        self._state = NumberGuessState()

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task: Optional[dict] = None, **kwargs) -> NumberGuessObservation:
        if seed is not None:
            random.seed(seed)
        difficulty: TaskDifficulty = task.get("difficulty", "medium") if task else "medium"
        if difficulty not in self.TASK_CONFIG:
            difficulty = "medium"

        cfg = self.TASK_CONFIG[difficulty]
        self._secret = random.randint(cfg["min"], cfg["max"])
        self._remaining = cfg["max_attempts"]
        self._difficulty = difficulty
        self._state = NumberGuessState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            max_attempts=cfg["max_attempts"],
            secret_number=self._secret,
            difficulty=difficulty,
        )
        return NumberGuessObservation(
            done=False,
            reward=0.0,
            feedback="",
            attempts_remaining=self._remaining,
            message=f"[{difficulty.upper()}] Guess a number between {cfg['min']}–{cfg['max']}! ({self._remaining} attempts)",
            progress_score=0.0,
        )

    def step(self, action: NumberGuessAction, **kwargs) -> NumberGuessObservation:
        self._state.step_count += 1
        guess = action.guess
        cfg = self.TASK_CONFIG[self._difficulty]

        if not (cfg["min"] <= guess <= cfg["max"]):
            self._remaining -= 1
            return NumberGuessObservation(
                done=self._remaining <= 0,
                reward=-0.1,
                feedback="invalid",
                attempts_remaining=self._remaining,
                message="Guess out of range!",
                progress_score=0.0,
            )

        self._remaining -= 1
        distance = abs(guess - self._secret)
        max_dist = cfg["max"] - cfg["min"]

        if guess < self._secret:
            feedback = "too low"
            reward = -0.01 + (1 - distance / max_dist) * 0.05   # partial progress
            message = f"{guess} is too low!"
        elif guess > self._secret:
            feedback = "too high"
            reward = -0.01 + (1 - distance / max_dist) * 0.05
            message = f"{guess} is too high!"
        else:
            feedback = "correct"
            reward = 1.0
            message = f"✅ Correct! Secret was {self._secret}."

        done = (feedback == "correct") or (self._remaining <= 0)

        # Final grader score (0.0–1.0)
        final_grade = None
        if done:
            if feedback == "correct":
                efficiency = (self._state.max_attempts - self._remaining + 1) / self._state.max_attempts
                final_grade = 0.7 + 0.3 * (1 - efficiency)   # 0.7 base + bonus for speed
            else:
                # closeness on failure
                final_grade = max(0.0, 1.0 - (distance / max_dist))

        return NumberGuessObservation(
            done=done,
            reward=reward,
            feedback=feedback,
            attempts_remaining=self._remaining,
            message=message,
            progress_score=max(0.0, reward),   # shaping signal
            final_grade=final_grade,
        )

    @property
    def state(self) -> NumberGuessState:
        return self._state
