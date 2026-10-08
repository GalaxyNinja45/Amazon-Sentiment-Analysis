"""Shared constants and helpers used by every script in the project."""
import re
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# The label sets. Edit TOPICS if you want to track different themes,
# then re-run extraction (and re-label your validation set).
TOPICS = ["taste", "price", "packaging", "nutrition_health", "shipping", "freshness_quality"]
SENTIMENTS = ["positive", "neutral", "negative"]


def score_to_sentiment(score) -> str:
    """Map a 1-5 star rating to a sentiment label (our 'free' ground truth).

    Caveat: star ratings are a noisy proxy. A 4-star review can still contain
    complaints, and a 3-star review can be clearly positive or negative.
    Reading the disagreements is part of the project (see evaluate.py).
    """
    score = int(score)
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def clean_text(s) -> str:
    """Strip HTML tags (the dataset contains <br />) and collapse whitespace."""
    s = re.sub(r"<[^>]+>", " ", str(s))
    return re.sub(r"\s+", " ", s).strip()
