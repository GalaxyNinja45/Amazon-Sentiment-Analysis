"""Step 4: turn the LLM output into something a business could act on.

Usage:
    python insights.py --variant v3_fewshot

Writes to outputs/: topic_by_sentiment.png, complaint_topics.csv, sample_complaints.csv

IMPORTANT: the sample is balanced by star rating, so it over-represents negative reviews
compared with the real dataset. Do NOT report overall sentiment shares as if they describe
all customers. Comparing topics *between* positive and negative reviews is still meaningful.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUT_DIR, TOPICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sample", default="data/sample.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    suffix = "_dryrun" if args.dry_run else ""

    sample = pd.read_csv(args.sample)
    with open(OUT_DIR / f"results_{args.variant}{suffix}.jsonl") as f:
        res = pd.DataFrame([json.loads(l) for l in f if l.strip()]).drop_duplicates("review_id", keep="last")
    df = sample.merge(res, on="review_id", how="inner").dropna(subset=["sentiment"])
    df = df[df["sentiment"].isin(["positive", "negative"])]

    # Share of reviews in each sentiment group that mention each topic.
    rates = {}
    for sent, g in df.groupby("sentiment"):
        rates[sent] = {t: g["topics"].map(lambda ts: t in ts).mean() for t in TOPICS}
    rates = pd.DataFrame(rates).sort_values("negative", ascending=False)
    rates.to_csv(OUT_DIR / "complaint_topics.csv")
    print((rates * 100).round(1).to_string())

    ax = rates.plot.bar(figsize=(8, 4.5), color={"negative": "#c0392b", "positive": "#2e86c1"})
    ax.set_ylabel("Share of reviews mentioning topic")
    ax.set_title("What do positive vs. negative reviewers talk about?")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "topic_by_sentiment.png", dpi=150)

    # Readable examples of what customers actually complain about, per topic.
    neg = df[(df["sentiment"] == "negative") & (df["complaint_summary"].fillna("") != "")]
    rows = []
    for t in TOPICS:
        for _, r in neg[neg["topics"].map(lambda ts: t in ts)].head(3).iterrows():
            rows.append({"topic": t, "complaint_summary": r["complaint_summary"], "review_id": r["review_id"]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "sample_complaints.csv", index=False)
    print(f"\nSaved charts and tables to {OUT_DIR}/")


if __name__ == "__main__":
    main()
