import nltk
from collections import Counter

# Sample reviews
reviews = [
    'Great battery life and camera quality',
    'Overheats sometimes and storage is limited',
    'Smooth performance but lacks fast charging'
]

# Tokenize and extract keywords
all_words = []
for review in reviews:
    tokens = nltk.word_tokenize(review.lower())
    all_words.extend(tokens)

# Count most common words
word_freq = Counter(all_words)
print(word_freq.most_common(10))
