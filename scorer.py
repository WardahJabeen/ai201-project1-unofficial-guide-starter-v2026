"""
Deciding whether an answer was right.

`run_eval.py` looks for this file and calls `judge` once per run. If it isn't
here, the Run columns in your run log come out blank and you read the answers
yourself. With it here, they carry verdicts.

⚠️ THIS IS A TEMPLATE. The numbers in it are starting points, not answers.

What counts as correct is the judgment this milestone is actually about, and
it is yours to make. Run `python scorer.py` to see how the current settings
score a handful of cases, then move SIMILARITY until the verdicts match what
you would have said reading the answers by hand. Write down what you changed
and why — that reasoning is worth more than the number.
"""

import re

from rapidfuzz import fuzz

# The refusal string, copied from gate.py. It is not imported from there on
# purpose: `import gate` pulls in store.py and with it the whole vector store,
# and comparing two strings should not need a database or an embedding model.
# Keeping the scorer importable on its own means `python scorer.py` still runs
# when something further down the pipeline is broken.
REFUSAL = "I don't have enough information about that."

# How close `expects` has to be to something in the answer, out of 100.
#
# Below about 70 almost anything matches and the scorer stops telling you
# anything. Above about 90 you are demanding near-exact wording, which mostly
# measures how chatty the model was feeling. 80 is a reasonable place to start
# arguing from.
SIMILARITY = 80

# Require every number in `expects` to appear in the answer.
#
# This exists because fuzzy matching is bad at exactly the thing this corpus
# is full of. "50 minutes" and "70 minutes" score 91 against each other —
# comfortably over any sane threshold — while meaning different things. Most
# of your questions turn on a number, so a near-miss on the digits is a wrong
# answer, not a close one.
CHECK_NUMBERS = True

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_PUNCTUATION = re.compile(r"[^\w\s]")


def normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Without this, "four minutes." and "four minutes" score differently for no
    reason anyone cares about.
    """
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def numbers_in(text: str) -> set[str]:
    """Every number in a string, normalised so 1,500 and 1500 are the same."""
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def similarity(expects: str, answer: str) -> float:
    """
    How well `expects` is covered by `answer`, out of 100.

    Two measures, because they fail in different places:

      - `partial_ratio` looks for the best matching window of the answer. This
        is the one that matters most here: `expects` is a short phrase and the
        answer is two or three sentences around it.
      - `token_set_ratio` ignores word order and duplicates, so it still scores
        well when the model says the right thing the other way round.

    Taking the higher of the two is deliberately generous. A scorer that is
    too strict quietly turns correct answers into failures, and you find out
    only by reading every answer anyway — which is the work you were trying
    to avoid.
    """
    expects, answer = normalise(expects), normalise(answer)
    if not expects:
        return 0.0
    return max(
        fuzz.partial_ratio(expects, answer),
        fuzz.token_set_ratio(expects, answer),
    )


def judge(question, expects, answer, results) -> bool:
    """
    Did this answer contain what you said you expected?

    Args:
        question: what was asked. Unused here — it is in the signature because
                  a stricter scorer may want it, and changing the signature
                  later means changing run_eval.py too.
        expects:  the phrase from questions.py you decided meant "correct",
                  written before you saw any output.
        answer:   what the system actually produced.
        results:  the retrieved chunks, nearest first. Unused here. Reach for
                  it if you want to score attribution — whether the file the
                  answer cites is one retrieval actually supplied.

    Returns True if the answer counts as correct.
    """
    if not expects or not expects.strip():
        return False

    # A refusal is never a correct answer to a question you expected an answer
    # to. It is criterion 3's business, and run_eval.py records it separately.
    if answer.strip() == REFUSAL.strip():
        return False

    if CHECK_NUMBERS:
        wanted = numbers_in(expects)
        if wanted and not wanted.issubset(numbers_in(answer)):
            return False

    return similarity(expects, answer) >= SIMILARITY


if __name__ == "__main__":
    # Calibration cases. No API calls — these are written out so you can move
    # SIMILARITY and watch the verdicts change.
    #
    # The last column is what you would say reading it. Where the scorer
    # disagrees with you, one of the two needs to change.
    cases = [
        ("Add four minutes.",
         "Add four minutes to any Brightwater walking estimate in winter "
         "(guide_walking.md).", True),
        ("From late September, when the term starts.",
         "Brightwater is at its busiest from late September as term starts, "
         "and accommodation becomes hard to find (guide_brightwater.md).", True),
        ("Marchwood's covered market.",
         "The covered market in Marchwood runs six days a week and is best on "
         "a weekday morning (guide_eating.md).", True),
        ("70 minutes",
         "The train takes 50 minutes from the regional hub (guide_brightwater.md).",
         False),
        ("On Fell Street, where prices are roughly half those on the harbour front.",
         REFUSAL, False),
        ("Add four minutes.",
         "Paths are cleared by 7am on weekdays (guide_walking.md).", False),
    ]

    print(f"SIMILARITY = {SIMILARITY}, CHECK_NUMBERS = {CHECK_NUMBERS}\n")
    print(f"{'score':>6}  {'scorer':<7} {'you':<7} expects")
    print("-" * 72)

    disagreements = 0
    for expects, answer, you_would_say in cases:
        verdict = judge("", expects, answer, [])
        if verdict != you_would_say:
            disagreements += 1
        print(
            f"{similarity(expects, answer):>6.0f}  "
            f"{'pass' if verdict else 'fail':<7} "
            f"{'pass' if you_would_say else 'fail':<7} "
            f"{expects[:44]}"
            f"{'   <-- disagrees' if verdict != you_would_say else ''}"
        )

    print()
    print(
        f"{len(cases) - disagreements} of {len(cases)} match your judgement."
        if disagreements
        else f"All {len(cases)} match your judgement at these settings."
    )
