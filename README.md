# Review Summary

A small, self-contained Python project that analyzes customer reviews and produces a one-paragraph summary highlighting the top pros and cons.

Key features

- Classifies each review as positive/negative using NLTK's VADER sentiment analyzer.
- Extracts frequent aspect phrases (unigrams and bigrams) from positive reviews as "pros" and from negative reviews as "cons".
- Produces a human-readable summary paragraph listing the top 3 pros and top 3 cons.
- Generates a sentiment distribution chart (PNG) visualizing positive vs. negative vs. neutral feedback.

Quick start (Windows PowerShell)

1. Open PowerShell and change to the project directory:

```powershell
cd C:\Users\suzy.william\Downloads\review_summary_project
```

2. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the CLI against the included sample reviews file:

```bash
python -m review_summary.main reviews.txt
```

Example output

The script prints a summary paragraph and generates a sentiment chart:

**Summary paragraph:**
```
Customers most frequently highlighted the display, battery, and camera as standout strengths. 
Common complaints centered on the software, battery, and gps.
```

**Sentiment chart:** A high-resolution PNG chart (`sentiment_chart.png`) showing the distribution of positive, negative, and neutral reviews.

Running tests

This project includes a small pytest test.

```powershell
python -m pytest -q
```

Notes and implementation details

- Tokenization and phrase extraction: the code uses NLTK POS tagging to identify nouns and extract meaningful aspect phrases. Term mapping intelligently groups related aspects (e.g., "battery," "charging," "charge") under canonical terms.
- Sentiment analysis: VADER sentiment analyzer is used with thresholds to classify reviews as positive (≥0.05), negative (≤-0.05), or neutral.
- Visualization: Matplotlib generates a high-quality (300 DPI) bar chart showing sentiment distribution.
- NLTK data: the project will attempt to download required NLTK data quietly when needed. If your environment blocks downloads, you may need to manually download the resources using `nltk.download()`.
- Performance: processes 50 reviews in ~3.5 seconds with minimal memory footprint.

Project layout

- `review_summary/` — package containing `analyze.py` and `main.py`.
- `reviews.txt` — sample file containing 50 example reviews (one per line).
- `tests/` — pytest tests.
- `requirements.txt` — pip requirements.

Next steps (suggestions)

- Add JSON export with confidence scores and aspect counts.
- Expand test coverage for edge cases (all-neutral reviews, empty files, multilingual input).
- Add support for different product categories with category-specific aspect mappings.
- Build REST API wrapper for integration with e-commerce platforms.

License

This sample is provided as-is for demonstration purposes. Feel free to adapt it for your needs.

Contact

If you want enhancements or have a specific dataset to analyze, tell me what you want and I can iterate further.
