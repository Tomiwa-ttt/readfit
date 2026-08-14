"""
ReadFit - linguistic feature extraction.
OWNER: Ameer Hamza Manzoor (Track 3)

Pure-python + numpy. No NLTK, no downloads, no network at runtime.
This matters: the GCP VM must not need to download anything on boot.
"""
import re
import numpy as np

VOWELS = "aeiouy"

# Dale-Chall style "easy words" - a short high-frequency list.
# Not the full 3000-word list; enough to make the feature informative.
COMMON_WORDS = set("""
a about after all also an and any are as at back be because been before big but
by call can come could day did do down each even find first for from get give go
good great had has have he her here him his how i if in into is it its just know
like little long look made make man many may me more most much must my never new
no not now of on one only or other our out over people put said same say see she
should so some take than that the their them then there these they thing think this
those time to too two up us use very want was way we well went were what when where
which who will with work would year you your
""".split())


def syllables(word):
    """Vowel-group heuristic. Not perfect; consistent, which is what a feature needs."""
    word = word.lower()
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def tokenize(text):
    return re.findall(r"[A-Za-z']+", text)


def sentences(text):
    parts = re.split(r"[.!?]+", text)
    return [p for p in parts if p.strip()]


def flesch_reading_ease(text):
    w = tokenize(text)
    s = sentences(text)
    if not w or not s:
        return 0.0
    syl = sum(syllables(x) for x in w)
    return 206.835 - 1.015 * (len(w) / len(s)) - 84.6 * (syl / len(w))


def flesch_kincaid_grade(text):
    w = tokenize(text)
    s = sentences(text)
    if not w or not s:
        return 0.0
    syl = sum(syllables(x) for x in w)
    return 0.39 * (len(w) / len(s)) + 11.8 * (syl / len(w)) - 15.59


LINGUISTIC_NAMES = [
    "word_count", "sentence_count", "mean_sentence_len", "max_sentence_len",
    "mean_word_len", "mean_syllables", "pct_polysyllabic", "pct_long_words",
    "type_token_ratio", "pct_common_words", "comma_rate", "flesch_reading_ease",
    "flesch_kincaid_grade", "pct_capitalised", "apostrophe_rate",
]


def linguistic_features(text):
    """Returns a fixed-length vector in LINGUISTIC_NAMES order."""
    w = tokenize(text)
    s = sentences(text)
    if not w:
        return np.zeros(len(LINGUISTIC_NAMES), dtype=float)

    lens = [len(x) for x in w]
    syl = [syllables(x) for x in w]
    slens = [len(tokenize(x)) for x in s] or [0]
    lower = [x.lower() for x in w]

    return np.array([
        len(w),
        len(s),
        float(np.mean(slens)),
        float(np.max(slens)),
        float(np.mean(lens)),
        float(np.mean(syl)),
        sum(1 for x in syl if x >= 3) / len(w),
        sum(1 for x in lens if x >= 7) / len(w),
        len(set(lower)) / len(w),
        sum(1 for x in lower if x in COMMON_WORDS) / len(w),
        text.count(",") / max(len(s), 1),
        flesch_reading_ease(text),
        flesch_kincaid_grade(text),
        sum(1 for x in w if x[0].isupper()) / len(w),
        text.count("'") / len(w),
    ], dtype=float)


def linguistic_matrix(texts):
    return np.vstack([linguistic_features(t) for t in texts])


# BT_easiness is a z-score-like scale: higher = easier. Bands calibrated on the
# CLEAR training distribution in train.py and written into metrics.json.
def band_from_score(score, cuts):
    """cuts = [c1, c2, c3, c4] ascending. Returns a human-readable band."""
    labels = ["Grades 11-12", "Grades 9-10", "Grades 6-8", "Grades 4-5", "Grades 2-3"]
    idx = 0
    for c in cuts:
        if score >= c:
            idx += 1
    return labels[min(idx, len(labels) - 1)]
