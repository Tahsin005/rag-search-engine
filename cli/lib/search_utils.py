import json
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)

CACHE_DIR = "cache"
DATA_DIR = "data"
MOVIES_PATH = f"{DATA_DIR}/movies.json"
STOPWORDS_PATH = f"{DATA_DIR}/stopwords.txt"

BM25_K1 = 1.5
BM25_B = 0.75


def preprocess(text: str) -> str:
    return text.lower().translate(TRANSLATION_TABLE)


def load_stopwords() -> set[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return {preprocess(word) for word in f.read().splitlines()}


STOP_WORDS = load_stopwords()


def tokenize(text: str) -> list[str]:
    tokens = preprocess(text).split()
    return [
        stemmer.stem(token)
        for token in tokens
        if token and token not in STOP_WORDS
    ]


def tokenize_term(term: str) -> str:
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token from '{term}', got {len(tokens)}")
    return tokens[0]


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data.get("movies", [])