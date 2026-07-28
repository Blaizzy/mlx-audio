from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class WERResult:
    substitutions: int
    deletions: int
    insertions: int
    reference_tokens: int
    hypothesis_tokens: int
    wer: float

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def to_dict(self) -> dict:
        data = asdict(self)
        data["edits"] = self.edits
        return data


def _prefer(candidate: tuple[int, int, int, int], best: tuple[int, int, int, int]):
    """Tie-break edit paths deterministically: fewer edits, then fewer insertions."""
    return (candidate[0], candidate[3], candidate[2], candidate[1]) < (
        best[0],
        best[3],
        best[2],
        best[1],
    )


def compute_wer(
    reference: str | Iterable[str], hypothesis: str | Iterable[str]
) -> WERResult:
    """Compute WER and edit operation counts.

    Args:
        reference: Reference text or pre-tokenized reference words.
        hypothesis: Hypothesis text or pre-tokenized hypothesis words.
    """
    ref_tokens = reference.split() if isinstance(reference, str) else list(reference)
    hyp_tokens = hypothesis.split() if isinstance(hypothesis, str) else list(hypothesis)

    n = len(ref_tokens)
    m = len(hyp_tokens)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        cost, subs, dels, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, subs, dels + 1, ins)

    for j in range(1, m + 1):
        cost, subs, dels, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, subs, dels, ins + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                best = dp[i - 1][j - 1]
            else:
                cost, subs, dels, ins = dp[i - 1][j - 1]
                best = (cost + 1, subs + 1, dels, ins)

            cost, subs, dels, ins = dp[i - 1][j]
            deletion = (cost + 1, subs, dels + 1, ins)
            if _prefer(deletion, best):
                best = deletion

            cost, subs, dels, ins = dp[i][j - 1]
            insertion = (cost + 1, subs, dels, ins + 1)
            if _prefer(insertion, best):
                best = insertion

            dp[i][j] = best

    edits, substitutions, deletions, insertions = dp[n][m]
    if n == 0:
        wer = 0.0 if edits == 0 else 1.0
    else:
        wer = edits / n

    return WERResult(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_tokens=n,
        hypothesis_tokens=m,
        wer=wer,
    )


def aggregate_wer(results: Iterable[WERResult]) -> dict:
    results = list(results)
    total_ref = sum(result.reference_tokens for result in results)
    total_subs = sum(result.substitutions for result in results)
    total_dels = sum(result.deletions for result in results)
    total_ins = sum(result.insertions for result in results)
    total_edits = total_subs + total_dels + total_ins
    wer_micro = total_edits / total_ref if total_ref else 0.0
    wer_macro = sum(result.wer for result in results) / len(results) if results else 0.0

    return {
        "wer_micro": wer_micro,
        "wer_macro": wer_macro,
        "substitution_rate": total_subs / total_ref if total_ref else 0.0,
        "deletion_rate": total_dels / total_ref if total_ref else 0.0,
        "insertion_rate": total_ins / total_ref if total_ref else 0.0,
        "total_reference_tokens": total_ref,
        "total_hypothesis_tokens": sum(result.hypothesis_tokens for result in results),
        "total_substitutions": total_subs,
        "total_deletions": total_dels,
        "total_insertions": total_ins,
        "total_edits": total_edits,
    }
