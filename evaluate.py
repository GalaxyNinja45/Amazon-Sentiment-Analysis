"""Step 3: measure how well each prompt variant (and a classical baseline) performs.

Usage:
    python evaluate.py                                   # compares all variants that have results
    python evaluate.py --full Reviews.csv                # baseline trains on a large slice of the full dataset
    python evaluate.py --variants v1_basic v3_fewshot

Writes to outputs/: summary.csv, confusion_<name>.png, disagreements_<name>.csv
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from common import OUT_DIR, SENTIMENTS, TOPICS, clean_text, score_to_sentiment
from prompts import PROMPTS


def load_results(variant, suffix):
    path = OUT_DIR / f"results_{variant}{suffix}.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    return df.drop_duplicates("review_id", keep="last")  # a later success replaces an earlier error


def to_topic_set(value):
    if isinstance(value, list):
        return set(value)
    if isinstance(value, str):
        return {t.strip().lower() for t in value.split(";") if t.strip() and t.strip().lower() != "none"}
    return set()  # NaN for failed rows


def make_baseline():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def baseline_predictions(sample, full_path, train_n, seed):
    """TF-IDF + logistic regression sentiment baseline.

    Without --full: 5-fold cross-validation on the sample itself (only ~400 training rows per fold,
    so the baseline is handicapped). With --full: train on a large random slice of the rest of the
    dataset, excluding the evaluation reviews and any review with identical text.
    """
    text = sample["Summary"].fillna("") + " . " + sample["Text"]
    if not full_path:
        return cross_val_predict(make_baseline(), text, sample["true_sentiment"], cv=5)

    full = pd.read_csv(full_path, usecols=["Id", "Score", "Summary", "Text"]).dropna(subset=["Text"])
    full["Text"] = full["Text"].map(clean_text)
    full["Summary"] = full["Summary"].fillna("").map(clean_text)
    full = full[~full["Id"].isin(sample["review_id"]) & ~full["Text"].isin(sample["Text"])]
    full = full.drop_duplicates("Text").sample(n=min(train_n, len(full)), random_state=seed)
    model = make_baseline().fit(full["Summary"] + " . " + full["Text"], full["Score"].map(score_to_sentiment))
    print(f"Baseline trained on {len(full):,} reviews from the full dataset")
    return model.predict(text)


def evaluate_sentiment(name, df, pred_col):
    y_true, y_pred = df["true_sentiment"], df[pred_col].fillna("unparsed")
    print(f"\n=== Sentiment: {name} ===")
    print(classification_report(y_true, y_pred, labels=SENTIMENTS, zero_division=0))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=SENTIMENTS, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Sentiment vs. star rating: {name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)

    wrong = df[y_true != y_pred][["review_id", "Score", "true_sentiment", pred_col, "Summary", "Text"]]
    wrong.to_csv(OUT_DIR / f"disagreements_{name}.csv", index=False)
    return {
        "sentiment_accuracy": accuracy_score(y_true, y_pred),
        "sentiment_macro_f1": f1_score(y_true, y_pred, labels=SENTIMENTS, average="macro", zero_division=0),
    }


def evaluate_topics(name, df, labels):
    merged = labels.merge(df[["review_id", "topics"]], on="review_id", how="inner")
    if merged.empty:
        return {}
    mlb = MultiLabelBinarizer(classes=TOPICS)
    y_true = mlb.fit_transform(merged["hand_topics"])
    y_pred = mlb.transform(merged["topics"].map(to_topic_set))
    print(f"\n=== Topics: {name}  (n={len(merged)} hand-labeled reviews) ===")
    print(classification_report(y_true, y_pred, target_names=TOPICS, zero_division=0))
    return {
        "topic_micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "topic_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "topic_n_labeled": len(merged),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample.csv")
    ap.add_argument("--labels", default="data/topic_labels.csv")
    ap.add_argument("--variants", nargs="+", default=list(PROMPTS))
    ap.add_argument("--full", default=None, help="Path to full Reviews.csv for a stronger baseline")
    ap.add_argument("--baseline-train-n", type=int, default=50000)
    ap.add_argument("--dry-run", action="store_true", help="Evaluate the *_dryrun result files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    suffix = "_dryrun" if args.dry_run else ""

    sample = pd.read_csv(args.sample)

    labels = pd.read_csv(args.labels, dtype=str).fillna("")
    labels = labels[labels["topics"].str.strip() != ""].copy()
    labels["review_id"] = labels["review_id"].astype(int)
    labels["hand_topics"] = labels["topics"].map(to_topic_set)
    labels = labels[["review_id", "hand_topics"]]  # avoid a column-name clash with the LLM's "topics"
    unknown = set().union(*labels["hand_topics"]) - set(TOPICS) if len(labels) else set()
    if unknown:
        print(f"WARNING: unknown topic names in your labels (ignored in scoring): {sorted(unknown)}")
    print(f"{len(labels)} hand-labeled reviews found")

    rows = []
    for variant in args.variants:
        res = load_results(variant, suffix)
        if res is None:
            print(f"(no results for {variant}, skipping)")
            continue
        df = sample.merge(res, on="review_id", how="left")
        metrics = {"name": variant, "sentiment_failures": int(df["sentiment"].isna().sum())}
        metrics.update(evaluate_sentiment(variant, df.rename(columns={"sentiment": "pred"}), "pred"))
        if len(labels):
            metrics.update(evaluate_topics(variant, df, labels))
        metrics["input_tokens"] = int(res.get("input_tokens", pd.Series(dtype=float)).sum())
        metrics["output_tokens"] = int(res.get("output_tokens", pd.Series(dtype=float)).sum())
        rows.append(metrics)

    base = sample.copy()
    base["pred"] = baseline_predictions(sample, args.full, args.baseline_train_n, args.seed)
    metrics = {"name": "tfidf_logreg_baseline", "sentiment_failures": 0}
    metrics.update(evaluate_sentiment("tfidf_logreg_baseline", base, "pred"))
    rows.append(metrics)

    summary = pd.DataFrame(rows).set_index("name")
    summary.to_csv(OUT_DIR / "summary.csv")
    print("\n=== SUMMARY (copy this table into your README) ===")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
