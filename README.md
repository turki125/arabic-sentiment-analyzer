# محلل المشاعر العربي | Arabic Sentiment Analyzer

A machine learning web app that classifies Arabic text as **positive** or **negative** in real time. Built and trained from scratch using a real-world Arabic restaurant review dataset.

🔗 **Live demo:** [Try it here](https://arabic-sentiment-analyzer-hfkq7rsrfpnccksj2lsdbl.streamlit.app/)

## Overview

This project takes raw, unstructured Arabic text (like customer reviews) and predicts its sentiment using a classic NLP pipeline: text cleaning → TF-IDF vectorization → Logistic Regression classification.

## Dataset

- **Source:** 8,364 real Arabic restaurant reviews (qaym.com)
- **Labels:** Binary sentiment (positive / negative)
- **Class distribution:** ~71% positive, ~29% negative (handled with `class_weight='balanced'`)

## Approach

1. **Text cleaning** — removed URLs, English text, numbers, and punctuation from raw Arabic reviews
2. **Feature extraction** — TF-IDF vectorization (top 5,000 terms)
3. **Model** — Logistic Regression with balanced class weights
4. **Evaluation** — 80/20 stratified train/test split

## Results

| Metric | Negative | Positive |
|---|---|---|
| Precision | 0.67 | 0.90 |
| Recall | 0.78 | 0.84 |
| F1-score | 0.72 | 0.87 |

**Overall accuracy: 82.43%**

## Known limitation

The model occasionally struggles with Gulf/Saudi dialect words that carry different meanings than in Modern Standard Arabic (MSA) — for example, "رهيب" formally means "terrifying" but is commonly used as slang for "awesome" in Gulf dialect. This kind of MSA-vs-dialect ambiguity is a well-known open challenge in Arabic NLP.

## Tech stack

- Python, scikit-learn (TF-IDF, Logistic Regression)
- Streamlit (web app + deployment)
- pandas, re (data cleaning)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---
Built by Turki
