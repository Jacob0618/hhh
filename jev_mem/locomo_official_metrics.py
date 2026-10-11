"""Independent transcription of the pinned LoCoMo metric definitions.

Source: snap-research/locomo, commit cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc,
task_eval/evaluation.py (SHA256 below). This is offline postprocessing only and
does not alter the primary reader or existing custom lexical scores.

No explicit LICENSE was found in the repository root or task_eval at that commit;
do not infer a redistribution license from the project's public availability.
The source snapshot is retained for local scholarly verification.
"""
from __future__ import annotations

import collections
import string
import regex as re

SOURCE_COMMIT = "cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc"
SOURCE_SHA256 = "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
SOURCE_URL = f"https://github.com/snap-research/locomo/blob/{SOURCE_COMMIT}/task_eval/evaluation.py"
LICENSE_STATUS = "No explicit license found at pinned root/task_eval; unspecified, not asserted permissive."


def normalize_official(text):
    text = str(text).replace(",", "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _stemmed_f1(prediction, reference):
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    prediction = [stemmer.stem(token) for token in normalize_official(prediction).split()]
    reference = [stemmer.stem(token) for token in normalize_official(reference).split()]
    overlap = sum((collections.Counter(prediction) & collections.Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(prediction), overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def official_locomo_f1(prediction, reference, category):
    """Match eval_question_answering's category branch, not a set-exact score.

    Category1 averages, over comma-separated reference pieces, the best F1 against
    any comma-separated prediction piece. Unmatched extra predicted pieces are
    not globally penalized by this formula; do not relabel it complete-set F1.
    """
    if not isinstance(prediction, str):
        raise TypeError("This adapter evaluates the string predictions produced by our reader")
    reference = str(reference)
    if int(category) == 3:
        reference = reference.split(";")[0].strip()
    if int(category) in (2, 3, 4):
        return _stemmed_f1(prediction, reference)
    if int(category) == 1:
        predicted_parts = [part.strip() for part in prediction.split(",")]
        reference_parts = [part.strip() for part in reference.split(",")]
        return sum(max(_stemmed_f1(part, answer) for part in predicted_parts)
                   for answer in reference_parts) / len(reference_parts)
    if int(category) == 5:
        return float("no information available" in prediction.lower() or "not mentioned" in prediction.lower())
    raise ValueError("Unexpected LoCoMo category")


def official_auxiliary_exact_match(prediction, reference):
    """Official helper's normalized token-set equality; not its QA main metric."""
    return float(set(normalize_official(prediction).split()) == set(normalize_official(reference).split()))
