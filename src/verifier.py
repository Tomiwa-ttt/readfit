"""
ReadFit - deterministic phonics decodability verifier.
OWNER: Diwakar Saini (Track 4)

Rule-based grapheme-phoneme correspondence (GPC) table. Greedy longest-match.
NO CMUdict, NO trained segmenter - deliberately simple so it is finishable today.
This component contains no model and cannot hallucinate. That separation is the
point: the thing judging correctness is pure Python.
"""
import re

VOWEL_LETTERS = set("aeiou")

# Ordered longest-first so greedy matching prefers digraphs/trigraphs.
GRAPHEMES = [
    "igh", "ough", "tch", "dge",
    "sh", "ch", "th", "wh", "ph", "ck", "ng", "qu",
    "ai", "ay", "ee", "ea", "ie", "oa", "oe", "oo", "ue", "ui",
    "ar", "er", "ir", "or", "ur", "aw", "au", "ow", "ou", "oi", "oy",
    "ll", "ss", "ff", "zz",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]

# UFLI-style checkpoints. A reconstruction, not the official document - say so
# on the slide. Each lesson INCLUDES everything from earlier lessons.
LESSONS = {
    10: {
        "gpcs": {"a", "i", "o", "m", "s", "t", "p", "n", "c", "d", "f", "g", "h", "b", "l", "r"},
        "sight": {"the", "a", "is", "to", "has", "of", "i"},
        "label": "Lesson 10 - short a/i/o + core consonants",
    },
    34: {
        "gpcs": {"a", "e", "i", "o", "u", "m", "s", "t", "p", "n", "c", "d", "f", "g", "h",
                 "b", "l", "r", "j", "k", "v", "w", "x", "y", "z", "sh", "ch", "th", "ck",
                 "ll", "ss", "ff", "zz", "ng"},
        "sight": {"the", "a", "is", "to", "has", "of", "i", "was", "said", "you", "they", "he", "she"},
        "label": "Lesson 34 - all short vowels + sh/ch/th digraphs",
    },
    60: {
        "gpcs": {"a", "e", "i", "o", "u", "m", "s", "t", "p", "n", "c", "d", "f", "g", "h",
                 "b", "l", "r", "j", "k", "v", "w", "x", "y", "z", "sh", "ch", "th", "ck",
                 "ll", "ss", "ff", "zz", "ng", "wh", "ph", "tch", "dge", "ai", "ay", "ee",
                 "ea", "oa", "oe", "igh", "ie"},
        "sight": {"the", "a", "is", "to", "has", "of", "i", "was", "said", "you", "they",
                  "he", "she", "we", "be", "are", "were", "there", "what", "who"},
        "label": "Lesson 60 - + vowel teams and long vowels",
    },
    98: {
        "gpcs": set(GRAPHEMES),
        "sight": {"the", "a", "is", "to", "has", "of", "i", "was", "said", "you", "they",
                  "he", "she", "we", "be", "are", "were", "there", "what", "who", "would",
                  "could", "should", "their", "people", "because"},
        "label": "Lesson 98 - full basic code",
    },
}


def segment(word):
    """Greedy longest-match split into graphemes. Returns list, or None if stuck."""
    w = word.lower().strip("'")
    out, i = [], 0
    # silent-e: treat trailing consonant+e as a split-digraph, drop the e
    silent_e = (len(w) > 3 and w.endswith("e") and w[-2] not in VOWEL_LETTERS
                and w[-3] in VOWEL_LETTERS)
    if silent_e:
        w = w[:-1]
    while i < len(w):
        for g in GRAPHEMES:
            if w.startswith(g, i):
                out.append(g)
                i += len(g)
                break
        else:
            return None
    return out


VOWEL_LETTERS = set("aeiou")


def check_word(word, lesson_spec):
    """Returns (is_legal, reason). Fails closed: unknown -> illegal."""
    w = word.lower().strip("'")
    if not w:
        return True, "empty"
    if w in lesson_spec["sight"]:
        return True, "sight word"
    gs = segment(w)
    if gs is None:
        return False, "cannot segment"
    untaught = [g for g in gs if g not in lesson_spec["gpcs"]]
    if untaught:
        return False, "untaught: " + ", ".join(sorted(set(untaught)))
    return True, "decodable"


def check(text, lesson):
    """Returns the decodability report for a passage at a given lesson."""
    spec = LESSONS.get(int(lesson)) or LESSONS[34]
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return {"lesson": int(lesson), "lesson_label": spec["label"],
                "total_words": 0, "legal_words": 0, "decodability": 0.0,
                "illegal_words": [], "top_violations": []}

    illegal, violations = [], {}
    legal = 0
    for w in words:
        ok, reason = check_word(w, spec)
        if ok:
            legal += 1
        else:
            illegal.append({"word": w, "reason": reason})
            if reason.startswith("untaught: "):
                for g in reason.replace("untaught: ", "").split(", "):
                    violations[g] = violations.get(g, 0) + 1

    seen, unique = set(), []
    for item in illegal:
        k = item["word"].lower()
        if k not in seen:
            seen.add(k)
            unique.append(item)

    return {
        "lesson": int(lesson),
        "lesson_label": spec["label"],
        "total_words": len(words),
        "legal_words": legal,
        "decodability": round(100.0 * legal / len(words), 1),
        "illegal_words": unique[:40],
        "top_violations": sorted(violations.items(), key=lambda x: -x[1])[:8],
    }


def available_lessons():
    return [{"lesson": k, "label": v["label"]} for k, v in sorted(LESSONS.items())]
