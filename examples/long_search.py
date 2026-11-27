"""Select relevant report passages for a free-text lookup and build a Riley request."""

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    source: str
    text: str


def terms(text):
    return re.findall(r"[^\W_]+", text.casefold())


def retrieve(question, passages, *, limit=8):
    """Rank the authorized corpus by BM25, retaining complete passages and order.

    Use this for evidence lookup. Questions requiring the whole corpus, such
    as global extrema and counts, should use the unfiltered request instead.
    """
    if limit < 1:
        raise ValueError("passage limit must be positive")
    if len(passages) <= limit:
        return passages

    words = set(terms(question))
    matches = []
    frequency = Counter()
    total_length = 0
    for passage in passages:
        tokens = terms(passage.text)
        present = Counter(token for token in tokens if token in words)
        matches.append((len(tokens), present))
        frequency.update(present.keys())
        total_length += len(tokens)
    if not frequency:
        return passages

    weights = {
        word: math.log1p((len(passages) - count + 0.5) / (count + 0.5))
        for word, count in frequency.items()
    }
    average_length = total_length / len(passages)
    scores = []
    for index, (length, present) in enumerate(matches):
        normalization = 1.2 * (0.25 + 0.75 * length / average_length)
        score = sum(
            weights[word] * count * 2.2 / (count + normalization)
            for word, count in present.items()
        )
        if score > 0:
            scores.append((-score, index))
    selected = sorted(index for _, index in sorted(scores)[:limit])
    return [passages[index] for index in selected]


def search_request(model, question, evidence, *, token_witness=False):
    sources = "\n".join(f"[{p.source}] {p.text}" for p in evidence)
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nReport excerpts:\n{sources}\n\n"
                    f"Question: {question}\n"
                    "Answer in short natural-language sentences using the excerpts. "
                    "Write amounts in digits and put the exact source ID in square brackets "
                    "immediately after each amount. Use one citation per finding. "
                    "If the excerpts do not establish a finding, say that it is missing."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 96,
        "stream": True,
        "stream_options": {"include_usage": True},
        **({"return_token_ids": True} if token_witness else {}),
    }
