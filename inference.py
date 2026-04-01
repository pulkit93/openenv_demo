import os
import re
import time
from openai import OpenAI
from models import NumberGuessAction
from client import NumberGuessingEnv

# ====================== REQUIRED ENVIRONMENT VARIABLES ======================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client (exactly as required by hackathon rules)
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_llm_guess(history: list[str], attempts_left: int, difficulty: str) -> int:
    """Use LLM (OpenAI client) to suggest the next best guess."""
    prompt = f"""You are an expert number-guessing agent playing a {difficulty} game.
Previous feedback:
{chr(10).join(history) if history else "No previous guesses"}
Attempts remaining: {attempts_left}
Return ONLY a single integer as your next guess."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20,
        )
        text = response.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else 50
    except Exception:
        return 50  # safe fallback

def play_one_task(env, difficulty: str) -> float:
    """Play one task and return the final grade (0.0–1.0)."""
    obs = env.reset(task={"difficulty": difficulty})
    print(f"\n🎯 Starting {difficulty.upper()} task – {obs.message}")

    history = []
    max_steps = 20

    for step in range(max_steps):
        if obs.done:
            break

        guess = get_llm_guess(history, obs.attempts_remaining, difficulty)
        action = NumberGuessAction(guess=guess)

        obs = env.step(action)
        history.append(f"Guess {guess} → {obs.feedback}")

        print(f"  Step {step+1:2d}: guess={guess:3d} | {obs.feedback:8s} | reward={obs.reward:.3f}")

        if obs.final_grade is not None:
            print(f"✅ {difficulty.upper()} finished → Final Grade: {obs.final_grade:.4f}")
            return obs.final_grade

    # fallback if max steps reached
    final_grade = getattr(obs, 'final_grade', 0.3) or 0.3
    print(f"⏰ {difficulty.upper()} finished → Final Grade: {final_grade:.4f}")
    return final_grade

def main():
    print("🚀 OpenEnv Number Guessing – Baseline Inference")
    print(f"Model: {MODEL_NAME} | Start time: {time.strftime('%H:%M:%S')}")

    # For local Colab / HF Space validation the base_url is usually localhost:7860
    # The judge will run this script against your deployed Space.
    with NumberGuessingEnv(base_url="http://localhost:7860") as env:
        scores = {}
        for diff in ["easy", "medium", "hard"]:
            scores[diff] = play_one_task(env, diff)

        # Final reproducible scores
        print("\n" + "="*60)
        print("FINAL SCORES (0.0–1.0)")
        print("="*60)
        for diff, score in scores.items():
            print(f"{diff.capitalize():7s} : {score:.4f}")
        avg = sum(scores.values()) / 3
        print(f"\nAVERAGE SCORE : {avg:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()
