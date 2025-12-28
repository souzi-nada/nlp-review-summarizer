# Review Summary

A small, self-contained Python project that analyzes customer reviews and produces a one-paragraph summary highlighting the top pros and cons.

Key features

- Classifies each review as positive/negative using NLTK's VADER sentiment analyzer.
- Extracts frequent aspect phrases (unigrams and bigrams) from positive reviews as "pros" and from negative reviews as "cons".
- Produces a human-readable summary paragraph listing the top 3 pros and top 3 cons.

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

The script prints a summary paragraph, for example:

"Users most frequently praised battery, camera and performance. Common complaints were overheating, charging and storage."

Running tests

This project includes a small pytest test.

```powershell
python -m pytest -q
```

Notes and implementation details

- Tokenization and phrase extraction: to keep the project lightweight and avoid heavy NLTK runtime resource requirements, the code uses a regex-based tokenizer and extracts unigrams and bigrams as candidate aspect phrases. This is fast and robust, but less linguistically precise than full POS-based chunking.
- NLTK data: the project will attempt to download required NLTK data quietly when needed (VADER lexicon for sentiment). If your environment blocks downloads, you may need to manually download the resources using `nltk.download()`.
- Extensibility: if you'd like more accurate aspect extraction we can reintroduce NLTK POS tagging and noun-chunking (this requires the averaged_perceptron_tagger and punkt resources).

Project layout

- `review_summary/` — package containing `analyze.py` and `main.py`.
- `reviews.txt` — sample file containing 50 example reviews (one per line).
- `tests/` — pytest tests.
- `requirements.txt` — pip requirements.

Next steps (suggestions)

- Improve aspect extraction to prefer multi-word noun phrases (requires POS tagger).
- Add an output mode that exports JSON with counts and confidence scores.
- Add more tests covering edge cases (all-neutral reviews, empty file, multilingual input).

License

This sample is provided as-is for demonstration purposes. Feel free to adapt it for your needs.

Contact

If you want enhancements or have a specific dataset to analyze, tell me what you want and I can iterate further.
