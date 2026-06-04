from __future__ import annotations

import re
from dataclasses import dataclass


REFUSAL = (
    "I cannot help decide whether a fungus is edible, safe to eat, medically useful, "
    "or appropriate for dosage or field identification. I can discuss academic traits, "
    "ecology, toxicology, and risks with source-backed context, and you should consult "
    "qualified local experts for real-world identification or safety decisions."
)


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str = ""
    refusal: str = REFUSAL


class SafetyPolicy:
    unsafe_patterns = [
        r"\bsafe to eat\b",
        r"\bcan i eat\b",
        r"\bshould i eat\b",
        r"\bis this edible\b",
        r"\bedible\??\b",
        r"\bdosage\b",
        r"\bdose\b",
        r"\bhow much .* take\b",
        r"\btreat .* with\b",
        r"\bmedical advice\b",
        r"\bidentify this\b",
        r"\bfield identification\b",
        r"\bwhat mushroom is this\b",
        r"\bpoisonous or safe\b",
    ]

    def __init__(self, mode: str = "strict") -> None:
        self.mode = mode
        self._compiled = [re.compile(pattern, re.IGNORECASE) for pattern in self.unsafe_patterns]

    def check(self, text: str) -> SafetyDecision:
        for pattern in self._compiled:
            if pattern.search(text or ""):
                return SafetyDecision(False, f"Matched safety boundary: {pattern.pattern}")
        return SafetyDecision(True)

    def validate_response(self, text: str) -> list[str]:
        errors: list[str] = []
        lowered = (text or "").lower()
        risky_claims = [
            "safe to eat",
            "you can eat",
            "recommended dose",
            "take this mushroom",
            "definitely edible",
            "field identification confirms",
        ]
        for claim in risky_claims:
            if claim in lowered:
                errors.append(f"Response contains unsafe claim: {claim!r}")
        return errors
