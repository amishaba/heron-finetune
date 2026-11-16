"""Regenerates the open_ended fixture: 600 questions with long, free-form answers and no consistent target
(spec 12: the agent must decline; there's nothing to grade automatically).

    .venv/bin/python examples/fixtures/open_ended/make.py
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPICS = ["our onboarding process", "the pricing change", "remote work", "the new office", "quarterly planning",
          "the hiring freeze", "customer feedback", "the product roadmap", "team rituals", "the reorg"]  # fmt: skip
OPENERS = ["Honestly,", "In my view,", "It depends, but", "From what I've seen,", "Short answer: it's complicated."]
BODIES = [
    "the biggest issue is that nobody owns the decision, so it drifts until someone escalates it",
    "we should run a small experiment first and only then decide whether it scales",
    "the people closest to the work already know the answer; we just don't ask them early enough",
    "I'd focus on the second-order effects, because the obvious costs are easy and the hidden ones aren't",
    "it worked well for the first few months and then quietly stopped mattering to anyone",
    "the data points one way and everyone's gut points the other, which is usually a sign we're missing context",
]


def main() -> None:
    rng = random.Random(5)
    with (HERE / "survey.jsonl").open("w") as f:
        for i in range(600):
            topic = rng.choice(TOPICS)
            answer = " ".join([rng.choice(OPENERS)] + rng.sample(BODIES, 3)) + f". (respondent {i})"
            row = {"id": i, "question": f"What do you think about {topic}?", "answer": answer}
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
