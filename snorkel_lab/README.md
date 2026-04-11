# Data Labeling with Snorkel — Yelp Review Sentiment

A supervision pipeline using [Snorkel](https://www.snorkel.org/) to classify **Yelp restaurant and business reviews** as positive or negative without hand-labeled training data.

## Dataset

| | |
|---|---|
| **Source** | [`yelp_polarity`](https://huggingface.co/datasets/yelp_polarity) via Hugging Face Datasets |
| **Domain** | Restaurant & business service reviews |
| **Classes** | Negative (0) · Positive (1) |
| **Sample used** | 4,000 train (balanced) · 1,000 test (balanced) |

The `yelp_polarity` dataset contains user reviews from Yelp with binary sentiment labels derived from star ratings (1–2 stars → negative, 3–5 stars → positive).

## Pipeline Overview

```
Raw Yelp reviews
      │
      ▼
Labeling Functions (9 LFs)
  ├── lf_positive_words       — food/service positive keywords
  ├── lf_negative_words       — food/service negative keywords
  ├── lf_star_rating          — star count mentioned in text
  ├── lf_negative_phrases     — "never coming back", "stay away", …
  ├── lf_positive_phrases     — "highly recommend", "must try", …
  ├── lf_exclamation          — 3+ exclamation marks → positive
  ├── lf_all_caps             — 2+ ALL-CAPS words → frustrated/negative
  ├── lf_review_length        — very short reviews with strong signals
  └── lf_textblob_polarity    — TextBlob polarity score (threshold-tuned)
      │
      ▼
Snorkel LabelModel  (learns LF reliability without ground truth)
      │
      ▼
Probabilistic labels → filter abstained rows
      │
      ▼
Logistic Regression on bag-of-bigrams (CountVectorizer, 10k features)
      │
      ▼
Data Augmentation (SpaCy + WordNet synonym replacement TFs)
      │
      ▼
Final classifier evaluation
```

## Results

| Model | Accuracy |
|---|---|
| Majority Vote (baseline) | ~72% |
| Snorkel LabelModel | ~74% |
| Logistic Regression (original weak labels) | ~77% |
| Logistic Regression (augmented) | ~83–85% |

## Key Design Choices

- **`lf_review_length`** — a novel LF exploiting the fact that very short Yelp reviews (< 20 words) tend to be strong, unambiguous opinions. Combined with a keyword check, this catches blunt complaints and punchy praise that longer-form LFs miss.
- **TextBlob thresholds tuned for Yelp** — Yelp reviews tend to be more neutral in phrasing than product reviews; we use `polarity > 0.25` (positive) and `polarity < -0.05` (negative) rather than the standard `0.3 / -0.1`.
- **Yelp-specific vocabulary** — keywords like `delicious`, `fresh`, `rude`, `filthy`, `overpriced`, `inedible` are tuned for the restaurant/service domain rather than product reviews.

## Requirements

```
snorkel>=0.10.0
datasets
textblob
spacy
nltk
scikit-learn
pandas
numpy
```

Install:
```bash
pip install snorkel datasets textblob spacy
python -m textblob.download_corpora
python -m spacy download en_core_web_sm
```

## Running

Open `yelpReviews.ipynb` in Colab and run all cells top to bottom.
