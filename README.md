# Turning Food Reviews into Structured Insight with an LLM

Classify unstructured customer reviews into sentiment and topics using an LLM API, then
**measure** how accurate it is against ground truth and against a classical ML baseline.
LLM used: Claude Haiku 4.5

## Problem
Companies receive large volumes of free-text customer feedback. This project extracts
structured fields (sentiment, topics, complaint summary) from reviews and evaluates
whether an LLM is actually better than a simple TF-IDF + logistic regression model.

## Data
[Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) (Kaggle, ~568K reviews, 1999-2012).
Check the license on the Kaggle page before reusing. Filtered to milk / shake related
reviews and sampled evenly across star ratings (n = 500).

## Method
1. `prepare_data.py`: filter, clean, and sample reviews.
2. `extract.py`: LLM returns JSON (sentiment, topics, complaint summary). Three prompt variants in `prompts.py`.
3. `evaluate.py`: sentiment scored against star ratings (1-2 negative, 3 neutral, 4-5 positive);
   topics scored against ___ reviews I labeled by hand; compared with a TF-IDF + logistic regression baseline.
4. `insights.py`: topic mention rates in positive vs. negative reviews.

## Results
| Model | Sentiment accuracy | Sentiment macro-F1 | Topic micro-F1 |
|:---:|:---:|:---:|:---:|
| Prompt v1 (basic) | 0.82 | 0.737 | 0.773 |
| Prompt v2 (definitions) | 0.828 | 0.755 | 0.793 |
| Prompt v3 (few-shot) | 0.82 | 0.741 | 0.777 |
| TF-IDF + LogReg baseline | 0.746 | 0.717 | n/a |

All three prompts ended up performing quite similarly, categorizing reviews correctly 82-83% of the time. However, the token usage of each prompt differs:

| Model | Input Tokens | Output Tokens |
|:---:|:---:|:---:|
| Prompt v1 (basic) | 143846 | 28764 |
| Prompt v2 (definitions) | 230346 | 27795 |
| Prompt v3 (few-shot) | 325346 | 23584 | 

Prompt v1 utilized a total of 172,610 tokens (143,846 input, 28,764 output);
Prompt v2 utilized a total of 258,141 tokens (230,346 input, 27,795 output);
Prompt v3 utilized a total of 348,930 tokens (325,346 input, 23,584 output). 
Considering the similar performance across all 3 prompts, Prompt v1 is the most effective choice, as it is significantly cheaper compared to the other two. 

Going through a sample of 51 reviews analyzed using Prompt v1, 23 were "model_errors", where the model predicted the sentiment incorrectly; 18 were "ambiguous", meaning that a resonable human could argue either way for a particular review's sentiment; and 10 were considered to be "label_noise", meaning the review give didn't particularly match the star rating all too well or the review was unrelated to the product itself.

Reviews that praised some aspects of a product while criticizing others often tripped up the model. Review 133141 is a typical case. The reviewer complained that the vanilla flavor was too mild for a vanilla coffee, but called it delicious and described the smell as wonderful. The reviewer gave it 4 stars, so the label is positive, yet the model classified it as neutral. The model appears to give the criticism more weight than the reviewer did, who seems to have treated it as a minor flaw in an otherwise enjoyable product.

The LLM outperformed the baseline across all categories. Therefore, the LLM is a strong tool that can be relied upon when performing market research.

## Limitations
- Star ratings are a noisy proxy for sentiment; many "errors" may be the label's fault.
- The sample is balanced by star rating, so sentiment shares do not describe the real customer base.
- Topic labels come from one annotator (me) on a small set.
- Reviews end in 2012 and are not specific to any one brand.

## Reproduce
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...            # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
python prepare_data.py --input Reviews.csv --n 500
# hand-label data/topic_labels.csv
python extract.py --variant v1_basic --limit 5     # sanity check first
python extract.py --variant v1_basic
python extract.py --variant v2_definitions
python extract.py --variant v3_fewshot
python evaluate.py --full Reviews.csv
python insights.py --variant v3_fewshot            # use your best variant
```
`--dry-run` on `extract.py`, `evaluate.py`, and `insights.py` runs the whole pipeline with a
keyword heuristic instead of the API, to test your setup for free.

## Next steps (ideas)
- Track prompt experiments with MLflow.
- Add a RAG step to answer questions about the reviews.
- Show the insights in a Power BI or Streamlit dashboard.
