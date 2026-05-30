import re


AR_DIACRITICS = re.compile(
    r'[\u064B-\u065F\u0670\u06D6-\u06ED]'
)


ARABIC_STOPWORDS = {
    "في",
    "من",
    "على",
    "الى",
    "إلى",
    "عن",
    "ما",
    "متى",
    "هل",
    "ثم",
    "او",
    "أو",
    "و",
    "يا",
    "هو",
    "هي"
}


def normalize(text: str):

    text = str(text)

    text = re.sub(AR_DIACRITICS, "", text)

    text = (
        text
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def tokenize(text: str):

    text = normalize(text)

    tokens = re.findall(
        r"[\w\u0600-\u06FF]+",
        text
    )

    tokens = [
        t for t in tokens
        if t not in ARABIC_STOPWORDS
        and len(t) > 1
    ]

    return tokens
