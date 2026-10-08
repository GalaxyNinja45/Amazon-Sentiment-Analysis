"""Step 1: build the evaluation sample from the full Kaggle CSV.

Usage:
    python prepare_data.py --input Reviews.csv --n 500

Outputs:
    data/sample.csv          the reviews we will send to the LLM (balanced by star rating)
    data/topic_labels.csv    a template for you to hand-label topics (never overwritten)
"""
import argparse
import re

import pandas as pd

from common import DATA_DIR, clean_text, score_to_sentiment

DEFAULT_KEYWORDS = ["milk", "protein shake", "protein drink", "shake"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Reviews.csv from Kaggle")
    ap.add_argument("--n", type=int, default=500, help="Total sample size (split evenly across 1-5 stars)")
    ap.add_argument("--label-n", type=int, default=100, help="How many reviews to include in the hand-labeling template")
    ap.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS, help="Keep reviews mentioning any of these")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input, usecols=["Id", "ProductId", "Score", "Time", "Summary", "Text"])
    print(f"Loaded {len(df):,} reviews")

    df = df.dropna(subset=["Text"]).copy()
    df["Text"] = df["Text"].map(clean_text)
    df["Summary"] = df["Summary"].fillna("").map(clean_text)
    # The dataset contains many repeated reviews (same text under different products).
    df = df.drop_duplicates(subset="Text")

    pattern = r"\b(?:" + "|".join(re.escape(k) for k in args.keywords) + r")\b"
    mask = df["Text"].str.contains(pattern, case=False, regex=True) | df["Summary"].str.contains(
        pattern, case=False, regex=True
    )
    df = df[mask]
    df = df[df["Text"].str.len().between(args.min_chars, args.max_chars)]
    print(f"{len(df):,} reviews match keywords {args.keywords} and the length filter")
    print("Available per star rating:")
    print(df["Score"].value_counts().sort_index().to_string())

    per_score = args.n // 5
    parts = [
        g.sample(n=min(per_score, len(g)), random_state=args.seed)
        for _, g in df.groupby("Score")
    ]
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample = sample.rename(columns={"Id": "review_id"})
    sample["true_sentiment"] = sample["Score"].map(score_to_sentiment)

    out = DATA_DIR / "sample.csv"
    sample.to_csv(out, index=False)
    print(f"\nSaved {len(sample)} reviews to {out}")

    labels_path = DATA_DIR / "topic_labels.csv"
    if labels_path.exists():
        print(f"{labels_path} already exists, leaving your hand labels untouched.")
    else:
        template = sample.head(args.label_n)[["review_id", "Summary", "Text"]].copy()
        template["topics"] = ""
        template.to_csv(labels_path, index=False)
        print(f"Wrote labeling template to {labels_path}")
        print("Fill the 'topics' column using a semicolon-separated list, e.g.  taste;price")
        print("Use 'none' if no topic applies. Leave blank to skip a row.")


if __name__ == "__main__":
    main()
