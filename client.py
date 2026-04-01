from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import NumberGuessAction, NumberGuessObservation, NumberGuessState

class NumberGuessingEnv(EnvClient[NumberGuessAction, NumberGuessObservation, NumberGuessState]):
    """OpenEnv client – works both locally and on HF Spaces."""

    def _step_payload(self, action: NumberGuessAction) -> dict:
        return {"guess": action.guess}

    def _parse_result(self, payload: dict) -> StepResult:
        obs = payload.get("observation", {})
        return StepResult(
            observation=NumberGuessObservation(
                done=payload.get("done", False),
                reward=payload.get("reward", 0.0),
                feedback=obs.get("feedback", ""),
                attempts_remaining=obs.get("attempts_remaining", 0),
                message=obs.get("message", ""),
                progress_score=obs.get("progress_score", 0.0),
                final_grade=obs.get("final_grade"),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> NumberGuessState:
        return NumberGuessState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            max_attempts=payload.get("max_attempts", 10),
            secret_number=payload.get("secret_number", 0),
            difficulty=payload.get("difficulty", "medium"),
        )
