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
        "punkt_tab",
        "stopwords",
        "vader_lexicon",
        "averaged_perceptron_tagger_eng",
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
    
    # Common sentiment adjectives and weak nouns to filter out
    sentiment_words = {
        "great", "excellent", "amazing", "fantastic", "good", "bad", "poor", "awful",
        "terrible", "wonderful", "nice", "awesome", "okay", "ok", "mediocre", "very",
        "fast", "slow", "much", "many", "more", "less", "fine", "long", "high", "low",
        "possible", "likely", "sure", "big", "small", "life", "day", "time", "thing",
        "way", "using", "use", "feel", "feeling"
    }
    
    aspects = []
    import re
    import nltk

    for text in texts:
        # Tokenize and tag parts of speech
        tokens = nltk.word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract nouns and noun phrases
        for i, (word, pos) in enumerate(pos_tags):
            # Include nouns (NN, NNS, NNP, NNPS)
            if pos in ("NN", "NNS", "NNP", "NNPS"):
                if (word not in stop and word not in sentiment_words and 
                    not word.isdigit() and len(word) > 2):
                    lemma = lemmatizer.lemmatize(word, pos="n")
                    aspects.append(lemma)
                    
                    # Also capture noun phrases (adjective + noun or noun + noun)
                    if i > 0:
                        prev_word, prev_pos = pos_tags[i-1]
                        # Look for descriptive adjectives or nouns that modify
                        if (prev_pos in ("JJ", "JJR", "JJS", "NN", "NNS") and 
                            prev_word not in sentiment_words and prev_word not in stop and
                            len(prev_word) > 2 and not prev_word.isdigit()):
                            prev_lemma = lemmatizer.lemmatize(prev_word, pos="a" if prev_pos in ("JJ", "JJR", "JJS") else "n")
                            phrase = f"{prev_lemma} {lemma}"
                            aspects.append(phrase)

    counts = Counter(aspects)
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
