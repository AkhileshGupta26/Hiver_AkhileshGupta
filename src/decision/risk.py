"""Risk assessment and sensitive inquiry detector.

Identifies security threats, credential compromise, fraudulent billing disputes,
legal/regulatory escalations, and explicit requests for human assistance.
"""

import re
from typing import Dict, List, Any

# Transparent rule patterns for high-risk customer support interactions
RISK_PATTERNS = {
    "EXPLICIT_HUMAN_REQUEST": [
        r"\b(speak|talk) to a (human|person|agent|representative|supervisor|manager)\b",
        r"\breal person\b",
        r"\bstop (the |your )?bot\b",
        r"\btransfer me\b",
        r"\bconnect me to (an? )?(agent|human)\b",
        r"\bget me a human\b"
    ],
    "SECURITY_ACCOUNT_COMPROMISE": [
        r"\bhacked\b",
        r"\bunauthorized access\b",
        r"\bsomeone changed my (password|email|number)\b",
        r"\baccount (taken over|compromised|breached)\b",
        r"\bstolen credentials\b",
        r"\bsuspicious activity\b"
    ],
    "FINANCIAL_FRAUD_DISPUTE": [
        r"\bunauthorized\b",
        r"\b(stolen|lost) (credit|debit)?\s*card\b",
        r"\bcard fraud\b",
        r"\bbank dispute\b",
        r"\bchargeback\b",
        r"\bidentity theft\b",
        r"\bdon'?t recognize\b"
    ],
    "LEGAL_OR_REGULATORY_THREAT": [
        r"\blawyer\b",
        r"\battorney\b",
        r"\bsue (you|amazon)\b",
        r"\blawsuit\b",
        r"\bconsumer protection\b",
        r"\bbetter business bureau\b|\bbbb\b",
        r"\bftc\b|\bpolice report\b"
    ],
    "SAFETY_OR_HAZARDOUS_DEFECT": [
        r"\b(caught|set) (on )?fire\b",
        r"\bexploded\b",
        r"\belectric shock\b",
        r"\bburn(ed|t)? my\b",
        r"\bchemical leak\b",
        r"\bhospital|injury|injured\b"
    ]
}

class RiskDetector:
    """Deterministic, rule-grounded risk and safety assessment."""

    def __init__(self):
        self.compiled_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in RISK_PATTERNS.items()
        }

    def assess_risk(self, message: str) -> Dict[str, Any]:
        """Assess risk categories, triggers, and overall risk tier."""
        clean_msg = message.strip()
        triggered_categories = []
        matched_terms = []

        for category, patterns in self.compiled_patterns.items():
            for p in patterns:
                m = p.search(clean_msg)
                if m:
                    triggered_categories.append(category)
                    matched_terms.append(m.group(0))
                    break

        is_high_risk = len(triggered_categories) > 0
        risk_tier = "CRITICAL" if any(c in ["SECURITY_ACCOUNT_COMPROMISE", "SAFETY_OR_HAZARDOUS_DEFECT", "LEGAL_OR_REGULATORY_THREAT"] for c in triggered_categories) \
                    else "HIGH" if is_high_risk \
                    else "LOW"

        risk_score = 1.0 if risk_tier == "CRITICAL" else 0.8 if risk_tier == "HIGH" else 0.1

        return {
            "risk_tier": risk_tier,
            "risk_score": risk_score,
            "is_high_risk": is_high_risk,
            "triggered_categories": triggered_categories,
            "matched_terms": matched_terms,
            "explicit_human_requested": "EXPLICIT_HUMAN_REQUEST" in triggered_categories
        }

if __name__ == "__main__":
    detector = RiskDetector()
    test_cases = [
        "Where is my package? It is 3 days late.",
        "Someone hacked my account and changed the email address!",
        "I demand to speak to a human agent right now.",
        "Your charger caught on fire and almost burned down my house.",
        "Seeing an unauthorized charge of $800 on my card that I never authorized."
    ]
    for msg in test_cases:
        res = detector.assess_risk(msg)
        print(f"\nMessage: {msg}")
        print(f"Risk Tier: {res['risk_tier']} | Categories: {res['triggered_categories']}")
