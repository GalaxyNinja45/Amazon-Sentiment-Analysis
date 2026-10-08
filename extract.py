"""Step 2: run an LLM over the sample and save structured output.

Usage:
    export ANTHROPIC_API_KEY=...            (macOS/Linux)
    python extract.py --variant v1_basic --limit 5      # try 5 reviews first!
    python extract.py --variant v1_basic                # full run
    python extract.py --variant v1_basic --dry-run      # no API calls; tests the pipeline for free

Results are appended to outputs/results_<variant>.jsonl. Re-running resumes where
it left off (and retries rows that previously failed).
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd

from common import OUT_DIR, SENTIMENTS, TOPICS
from prompts import PROMPTS

try:
    import anthropic
except ImportError:  # only needed for real runs
    anthropic = None

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # a small, inexpensive model is plenty for this task


def build_user_message(row) -> str:
    return f"Review title: {row['Summary']}\n\nReview text:\n{str(row['Text'])[:2000]}"


def parse_response(raw: str) -> dict:
    """Parse and validate the model's JSON. Raises ValueError if it is unusable."""
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        raise ValueError("no JSON object found")
    obj = json.loads(m.group(0))
    sentiment = str(obj.get("sentiment", "")).strip().lower()
    if sentiment not in SENTIMENTS:
        raise ValueError(f"invalid sentiment: {sentiment!r}")
    topics = obj.get("topics", [])
    if not isinstance(topics, list):
        raise ValueError("topics is not a list")
    topics = sorted({t.strip().lower() for t in topics if isinstance(t, str)} & set(TOPICS))
    summary = str(obj.get("complaint_summary") or "").strip()
    return {"sentiment": sentiment, "topics": topics, "complaint_summary": summary}


def call_model(client, model, system, user, max_retries=4):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                # anthropic SDK 1.0+ no longer accepts temperature= as a keyword, so we send it
                # in the request body. If the API ever rejects it (HTTP 400 mentioning
                # "temperature"), delete this line; results just become slightly less repeatable.
                extra_body={"temperature": 0},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parsed = parse_response(resp.content[0].text)
            return parsed, resp.usage.input_tokens, resp.usage.output_tokens
        except ValueError as e:  # includes json.JSONDecodeError (bad/unparseable model output)
            last_err = e
        except anthropic.APIStatusError as e:
            if e.status_code in (401, 403, 404):  # bad key, no permission, or unknown model: nothing will work
                sys.exit(f"Fatal API error ({e.status_code}): {e.message}")
            if e.status_code == 429 or e.status_code >= 500:  # rate limit or server trouble: worth retrying
                last_err = e
                time.sleep(2 ** attempt)
            else:  # other 4xx (e.g. 400): retrying the same request will not help
                raise RuntimeError(f"API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:  # network blip
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {max_retries} attempts: {last_err}")


def fake_extract(row):
    """Crude keyword heuristic used by --dry-run. NOT an LLM; only for testing the pipeline."""
    text = f"{row['Summary']} {row['Text']}".lower()
    neg = sum(w in text for w in ["bad", "awful", "terrible", "disappoint", "waste", "stale", "never again"])
    pos = sum(w in text for w in ["love", "great", "delicious", "excellent", "perfect", "best", "recommend"])
    sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
    keywords = {
        "taste": ["taste", "flavor"], "price": ["price", "expensive", "cost"],
        "packaging": ["package", "carton", "leak"], "nutrition_health": ["protein", "sugar", "calorie"],
        "shipping": ["shipping", "arrived", "delivery"], "freshness_quality": ["fresh", "stale", "expired"],
    }
    topics = [t for t, ws in keywords.items() if any(w in text for w in ws)]
    return {"sentiment": sentiment, "topics": topics, "complaint_summary": ""}, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(PROMPTS))
    ap.add_argument("--input", default="data/sample.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N reviews (for testing)")
    ap.add_argument("--dry-run", action="store_true", help="Skip the API and use a keyword heuristic")
    args = ap.parse_args()

    client = None
    if not args.dry_run:
        if anthropic is None:
            sys.exit("The 'anthropic' package is not installed. Run: pip install -r requirements.txt")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set the ANTHROPIC_API_KEY environment variable first (never hard-code the key).")
        client = anthropic.Anthropic()

    sample = pd.read_csv(args.input)
    if args.limit:
        sample = sample.head(args.limit)

    out_path = OUT_DIR / f"results_{args.variant}{'_dryrun' if args.dry_run else ''}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                if "error" not in rec:
                    done.add(rec["review_id"])
    todo = sample[~sample["review_id"].isin(done)]
    print(f"{len(done)} already done, {len(todo)} to process -> {out_path}")

    system = PROMPTS[args.variant]
    tok_in = tok_out = failures = consecutive = 0
    with open(out_path, "a") as f:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            rec = {"review_id": int(row["review_id"]), "variant": args.variant, "model": args.model}
            try:
                if args.dry_run:
                    parsed, ti, to = fake_extract(row)
                else:
                    parsed, ti, to = call_model(client, args.model, system, build_user_message(row))
                rec.update(parsed, input_tokens=ti, output_tokens=to)
                tok_in += ti
                tok_out += to
                consecutive = 0
            except RuntimeError as e:
                rec["error"] = str(e)
                failures += 1
                consecutive += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if consecutive >= 5:
                sys.exit(f"Stopping: 5 reviews in a row failed. Last error: {rec.get('error')}")
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} processed ({failures} failures)")

    print(f"\nDone. failures={failures}  input_tokens={tok_in:,}  output_tokens={tok_out:,}")
    print("Multiply token counts by your model's current per-token prices to get the cost.")


if __name__ == "__main__":
    main()