import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import string
from typing import List, Tuple, Dict


def _download_nltk_data():
    # Ensure required NLTK data is available. Download quietly if missing.
    resources = [
        "punkt",
        "stopwords",
        "vader_lexicon",
        "averaged_perceptron_tagger",
        "wordnet",
        "omw-1.4",
    ]
    for r in resources:
        # Use nltk.download which is safe to call repeatedly; set quiet=True to reduce output
        try:
            nltk.download(r, quiet=True)
        except Exception:
            # If download fails, ignore here and let the calling code surface a helpful LookupError later
            pass


def load_reviews(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def _extract_top_nouns(texts: List[str], top_n: int = 3) -> List[Tuple[str, int]]:
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words("english")) | set(string.punctuation)
    nouns = []
    import re

    for text in texts:
        # Use a simple regex tokenizer to avoid dependency on punkt and taggers
        tokens = re.findall(r"\b[\w']+\b", text.lower())

        # Filter tokens: remove stopwords, punctuation-only tokens and short tokens
        filtered_tokens = [t for t in tokens if t not in stop and not t.isdigit() and len(t) > 1]

        # Build candidate phrases (unigrams and bigrams) from filtered tokens
        for i in range(len(filtered_tokens)):
            # unigram
            nouns.append(filtered_tokens[i])
            # bigram
            if i + 1 < len(filtered_tokens):
                nouns.append(filtered_tokens[i] + " " + filtered_tokens[i + 1])

    # Filter out stopwords and short tokens
    filtered = [n for n in nouns if n and not all(ch.isdigit() for ch in n) and len(n) > 1]
    counts = Counter(filtered)
    return counts.most_common(top_n)


def analyze_reviews(reviews: List[str], top_n: int = 3) -> Dict[str, List[Tuple[str, int]]]:
    """Analyze reviews and return top pro and con nouns.

    Returns dict with keys: 'pros' and 'cons' each a list of (phrase, count).
    """
    _download_nltk_data()

    sia = SentimentIntensityAnalyzer()

    pos_texts = []
    neg_texts = []

    for r in reviews:
        if not r:
            continue
        scores = sia.polarity_scores(r)
        comp = scores.get("compound", 0.0)
        if comp >= 0.05:
            pos_texts.append(r)
        elif comp <= -0.05:
            neg_texts.append(r)
        else:
            # neutral reviews ignored for pros/cons extraction
            pass

    top_pros = _extract_top_nouns(pos_texts, top_n=top_n)
    top_cons = _extract_top_nouns(neg_texts, top_n=top_n)

    return {"pros": top_pros, "cons": top_cons}


def generate_summary(pros: List[Tuple[str, int]], cons: List[Tuple[str, int]]) -> str:
    """Generate a paragraph summarizing top pros and cons.

    pros/cons are lists of (phrase, count).
    """
    pros_list = [p for p, _ in pros]
    cons_list = [c for c, _ in cons]

    if not pros_list:
        pros_part = "No clear pros emerged from the reviews."
    else:
        pros_part = "Users most frequently praised " + ", ".join(pros_list[:-1]) + (" and " + pros_list[-1] if len(pros_list) > 1 else pros_list[0]) + "."

    if not cons_list:
        cons_part = "No clear cons were mentioned frequently."
    else:
        cons_part = "Common complaints were " + ", ".join(cons_list[:-1]) + (" and " + cons_list[-1] if len(cons_list) > 1 else cons_list[0]) + "."

    return f"{pros_part} {cons_part}"
