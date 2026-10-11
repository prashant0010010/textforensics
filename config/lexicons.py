"""Editable linguistic category lists.

These are *descriptive categories*, not lists of "AI words". Any marker can be
common in careful human writing; the analysers only measure how often and how
evenly categories are used within the document being analysed.
"""
from __future__ import annotations

PRONOUNS = frozenset("""i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his
himself she her hers herself it its itself they them their theirs themselves who whom whose which what this that
these those someone anyone everyone nobody somebody anybody everybody something anything everything nothing one
ones each other another""".split())

ARTICLES = frozenset(["a", "an", "the"])

PREPOSITIONS = frozenset("""about above across after against along among around as at before behind below beneath beside
between beyond by despite down during except for from in inside into like near of off on onto out outside over past
per since through throughout till to toward towards under underneath until up upon via with within without""".split())

CONJUNCTIONS = frozenset("""and but or nor so yet for although though because since unless while whereas whether if
once than that as both either neither""".split())

AUXILIARIES = frozenset("""am is are was were be been being have has had having do does did doing will would shall should
can could may might must ought""".split())

FUNCTION_WORDS = PRONOUNS | ARTICLES | PREPOSITIONS | CONJUNCTIONS | AUXILIARIES | frozenset(
    "not no very too also just only then there here when where why how all any some such more most much many few".split()
)

FW_CATEGORIES = {
    "pronouns": PRONOUNS,
    "articles": ARTICLES,
    "prepositions": PREPOSITIONS,
    "conjunctions": CONJUNCTIONS,
    "auxiliaries": AUXILIARIES,
}

DISCOURSE_CATEGORIES = {
    "additive": ["moreover", "furthermore", "in addition", "additionally", "also", "besides", "likewise",
                 "similarly", "equally", "what is more", "as well as"],
    "contrastive": ["however", "nevertheless", "nonetheless", "on the other hand", "in contrast", "conversely",
                    "although", "whereas", "yet", "despite this", "instead", "on the contrary", "even so",
                    "by contrast", "that said"],
    "causal": ["therefore", "thus", "consequently", "as a result", "hence", "because of this", "accordingly",
               "for this reason", "due to", "so that", "this means", "which means"],
    "conclusive": ["in conclusion", "in summary", "to summarize", "to summarise", "overall", "ultimately",
                   "in short", "to conclude", "in essence", "all in all", "in brief", "to sum up"],
    "sequential": ["first", "firstly", "second", "secondly", "third", "thirdly", "finally", "next", "then",
                   "subsequently", "meanwhile", "afterwards", "lastly", "initially", "to begin with"],
    "exemplification": ["for example", "for instance", "such as", "specifically", "in particular", "namely",
                        "e.g.", "to illustrate"],
}

HEDGES = ["may", "might", "could", "perhaps", "possibly", "probably", "likely", "unlikely", "seems", "seem",
          "appears to", "appear to", "suggests that", "generally", "typically", "often", "somewhat", "arguably",
          "tends to", "tend to", "to some extent", "it is possible", "potentially", "relatively", "in many cases",
          "to a certain degree", "can be", "usually", "mostly"]

QUALIFICATIONS = ["it is important to note", "it is worth noting", "it should be noted", "it is worth mentioning",
                  "it is essential to", "it is crucial to", "that being said", "with that said", "to be fair",
                  "needless to say", "keep in mind", "bear in mind", "to be clear", "in some cases",
                  "depending on", "this is not to say", "of course"]

# name -> regex. Editable; counted per 1000 words. Not AI fingerprints.
RHETORICAL_TEMPLATES = {
    "contrast-correction ('not just X, but Y')":
        r"\bnot (?:just|only|merely|simply)\b[^.?!;]{0,80}\b(?:but|it'?s|it is|also)\b",
    "denial-reframe ('it is not about X, it is about Y')":
        r"\b(?:it'?s|it is|this is|that'?s) not (?:just |simply )?(?:about|a matter of)\b[^.?!]{0,80}\b(?:it'?s|it is|but|rather)\b",
    "whether-or framing":
        r"\bwhether (?:you(?:'re| are)|it(?:'s| is)|they(?:'re| are)|we(?:'re| are))?[^.?!,]{0,50}\bor\b",
    "importance framing":
        r"\b(?:it is|it's) (?:important|worth|crucial|essential|vital) to (?:note|remember|recognize|recognise|consider|understand|mention)\b",
    "era-opening ('in today's ... world')":
        r"\bin (?:today'?s|the modern|an increasingly|the ever-?(?:changing|evolving))\b[^.?!]{0,40}\b(?:world|landscape|era|age|society|environment)\b",
    "'plays a crucial role'":
        r"\bplays? an? (?:\w+ )?(?:crucial|vital|key|significant|important|pivotal|central) role\b",
    "tricolon list (X, Y, and Z)":
        r"\b[\w'-]+, [\w'-]+,? (?:and|or) [\w'-]+\b",
}

CEE_EVIDENCE = [r"\bfor (?:example|instance)\b", r"\baccording to\b", r"\bstud(?:y|ies)\b", r"\bresearch\b",
                r"\bsuch as\b", r"\be\.g\.", r"\b\d{2,4}\b", r"\bevidence\b", r"\bdata\b", r"\bfound that\b"]
CEE_EXPLANATION = [r"\bthis (?:means|suggests|shows|indicates|demonstrates|implies|highlights)\b",
                   r"\btherefore\b", r"\bthus\b", r"\bas a result\b", r"\bwhich (?:means|shows|suggests|indicates)\b",
                   r"\bconsequently\b", r"\bin other words\b", r"\bthis is (?:because|why)\b"]

ABBREVIATIONS = frozenset("""mr mrs ms dr prof sr jr vs etc e.g i.e cf al fig figs no nos inc ltd co corp st mt approx eq eqs
ref refs vol vols pp p ch sec dept est u.s u.k a.m p.m jan feb mar apr jun jul aug sep sept oct nov dec""".split())

REFERENCE_HEADINGS = frozenset(["references", "bibliography", "works cited", "sources", "reference list",
                                "literature cited", "citations"])
