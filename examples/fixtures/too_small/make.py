"""Regenerates the too_small fixture: 60 labeled support messages (spec 12: the agent must decline).

.venv/bin/python examples/fixtures/too_small/make.py
"""

import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTENTS = {
    "lost_card": ["I lost my card", "my card is missing", "can't find my debit card"],
    "refund": ["I want a refund", "please refund my order", "money back for a cancelled order"],
    "change_address": ["update my address", "I moved, change my address", "new home address"],
    "close_account": ["close my account", "I want to cancel my account", "delete my account please"],
}


def main() -> None:
    rng = random.Random(3)
    rows = []
    for i in range(60):
        label = list(INTENTS)[i % len(INTENTS)]
        rows.append({"message": f"{rng.choice(INTENTS[label])} (#{100 + i})", "intent": label})
    with (HERE / "messages.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["message", "intent"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
