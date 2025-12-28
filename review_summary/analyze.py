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
        "way", "using", "use", "feel", "feeling", "issue", "problem", "aspect"
    }
    
    # Term mapping: group related aspects under canonical names
    term_mapping = {
        "battery": {"battery", "charging", "charge", "charger", "drain"},
        "camera": {"camera", "photo", "photography", "lens"},
        "display": {"display", "screen", "brightness", "bleeding", "color", "reproduction"},
        "performance": {"performance", "speed", "lag", "responsiveness", "multitasking"},
        "heating": {"heating", "overheat", "overheat", "warm", "temperature", "heat"},
        "build": {"build", "design", "casing", "quality", "weight", "solid", "premium", "material"},
        "audio": {"audio", "speaker", "sound", "microphone", "noise"},
        "fingerprint": {"fingerprint", "biometric", "recognition", "sensor"},
        "software": {"software", "firmware", "update", "ui", "app", "crash", "lag"},
        "connectivity": {"connectivity", "wifi", "bluetooth", "reception", "4g"},
        "storage": {"storage", "memory", "space"},
        "charger": {"charger", "charging"},
    }
    
    # Create reverse mapping for quick lookup
    reverse_mapping = {}
    for canonical, related in term_mapping.items():
        for term in related:
            reverse_mapping[term] = canonical
    
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
                    # Map to canonical term
                    canonical_term = reverse_mapping.get(lemma, lemma)
                    aspects.append(canonical_term)

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
    """Generate a narrative paragraph summarizing top pros and cons.

    pros/cons are lists of (phrase, count).
    Creates a cohesive summary paragraph highlighting the most frequently mentioned aspects.
    """
    pros_list = [p for p, _ in pros]
    cons_list = [c for c, _ in cons]

    # Build pros section
    if not pros_list:
        pros_part = "The reviews revealed no clear standout positive aspects."
    elif len(pros_list) == 1:
        pros_part = f"Users consistently praised the smartphone's {pros_list[0]}"
    elif len(pros_list) == 2:
        pros_part = f"The device was frequently commended for its {pros_list[0]} and {pros_list[1]}"
    else:
        pros_part = f"Customers most frequently highlighted the {pros_list[0]}, {pros_list[1]}, and {pros_list[2]} as standout strengths"

    # Build cons section
    if not cons_list:
        cons_part = "No major complaints were consistently mentioned."
    elif len(cons_list) == 1:
        cons_part = f"The primary concern noted was the phone's {cons_list[0]}"
    elif len(cons_list) == 2:
        cons_part = f"Main drawbacks included the {cons_list[0]} and {cons_list[1]}"
    else:
        cons_part = f"Common complaints centered on the {cons_list[0]}, {cons_list[1]}, and {cons_list[2]}"

    return f"{pros_part}. {cons_part}."
