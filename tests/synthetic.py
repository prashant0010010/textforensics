"""Seeded synthetic text generators used ONLY to exercise the mathematics and plumbing.

They say nothing about detector accuracy."""
from __future__ import annotations

import random
from typing import List

_NOUNS = "river market garden engine signal bridge window student forest theory pattern harbour lantern ledger compass meadow quarry statue orchard tunnel".split()
_VERBS = "carries shapes follows reveals borrows measures guards crosses repeats builds changes remembers".split()
_ADJ = "quiet ancient narrow bright distant fragile sudden careful hollow brittle patient restless".split()
_ADV = "slowly rarely often barely quickly gently carefully".split()
_FUNC = "the a of in on with from under over near between through during before after".split()


def make_sentence(rng: random.Random, n_words: int) -> str:
    words: List[str] = []
    while len(words) < n_words:
        words += [rng.choice(["the", "a", "this", "that"]), rng.choice(_ADJ), rng.choice(_NOUNS), rng.choice(_ADV),
                  rng.choice(_VERBS), rng.choice(_FUNC), rng.choice(["the", "a"]), rng.choice(_NOUNS)]
        if rng.random() < 0.3:
            words.insert(rng.randrange(len(words)), ",")
    words = words[:n_words]
    s = " ".join(words).replace(" ,", ",").strip(", ")
    return s[0].upper() + s[1:] + "."


def make_text(seed: int, n_words: int, para_words=(60, 160), sent_words=(6, 30), uniform: bool = False) -> str:
    rng = random.Random(seed)
    paras, total = [], 0
    while total < n_words:
        target = rng.randint(*para_words)
        sents, got = [], 0
        while got < target and total + got < n_words:
            k = 15 if uniform else rng.randint(*sent_words)
            sent = make_sentence(rng, k)
            sents.append(sent)
            got += len(sent.split())          # count real words so the target length is met
        paras.append(" ".join(sents))
        total += got
    return "\n\n".join(paras)


def repetitive_text(n_repeats: int = 40) -> str:
    return "\n\n".join(["In addition, the committee reviewed the proposal carefully. However, the proposal required further revision. "
                        "Therefore, the committee requested a second review of the proposal." for _ in range(n_repeats)])
