"""Multi-signal decision engine for automated triage, clarification, and escalation.

Evaluates intent classification confidence, dense retrieval evidence,
resolution consistency, and safety/security risk triggers to decide
between AUTO_HANDLE, ASK_CLARIFYING_QUESTION, and ESCALATE.
"""

from typing import Dict, List, Any, Optional

class DecisionEngine:
    """Multi-signal customer support decision and escalation engine."""

    def __init__(
        self,
        intent_confidence_threshold: float = 0.55,
        clarification_min_words: int = 3,
        auto_handle_min_evidence_score: float = 0.65
    ):
        self.conf_threshold = intent_confidence_threshold
        self.clarification_min_words = clarification_min_words
        self.auto_min_evidence = auto_handle_min_evidence_score

    def decide(
        self,
        message: str,
        predicted_intent: str,
        intent_confidence: float,
        risk_assessment: Dict[str, Any],
        evidence_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine triage action and output structured rationale."""
        reasons = []
        clean_words = message.strip().split()

        # Signal 1: Explicit human agent request
        if risk_assessment.get("explicit_human_requested"):
            reasons.append("Customer explicitly requested human assistance")
            return {
                "decision": "ESCALATE",
                "reasons": reasons,
                "confidence_score": intent_confidence,
                "action_code": "ESC_HUMAN_REQUEST"
            }

        # Signal 2: Security, account takeover, or legal risk
        if risk_assessment.get("is_high_risk"):
            categories = risk_assessment.get("triggered_categories", [])
            reasons.append(f"Security or policy risk detected: {', '.join(categories)}")
            return {
                "decision": "ESCALATE",
                "reasons": reasons,
                "confidence_score": intent_confidence,
                "action_code": "ESC_RISK_TRIGGER"
            }

        # Signal 3: Inherently sensitive intent category
        if predicted_intent in ["account_access_security", "payment_billing_issue"]:
            # If billing issue has low similarity or involves money dispute
            if predicted_intent == "account_access_security" or evidence_assessment.get("retrieval_confidence", 0) < 0.80:
                reasons.append(f"Inquiry belongs to security-sensitive intent category ('{predicted_intent}')")
                return {
                    "decision": "ESCALATE",
                    "reasons": reasons,
                    "confidence_score": intent_confidence,
                    "action_code": "ESC_SENSITIVE_INTENT"
                }

        # Signal 4: Contradictory / conflicting historical evidence
        if evidence_assessment.get("evidence_conflicts"):
            reasons.append("Historical agent resolutions conflict or lack operational consensus")
            return {
                "decision": "ESCALATE",
                "reasons": reasons,
                "confidence_score": intent_confidence,
                "action_code": "ESC_CONFLICTING_EVIDENCE"
            }

        # Signal 5: Insufficient evidence and low intent confidence
        evidence_sufficient = evidence_assessment.get("evidence_sufficient", False)
        retrieval_conf = evidence_assessment.get("retrieval_confidence", 0.0)

        if not evidence_sufficient and intent_confidence < self.conf_threshold:
            reasons.append(f"Low intent confidence ({intent_confidence:.2f} < {self.conf_threshold}) and insufficient evidence")
            return {
                "decision": "ESCALATE",
                "reasons": reasons,
                "confidence_score": intent_confidence,
                "action_code": "ESC_LOW_EVIDENCE_CONF"
            }

        if not evidence_sufficient and retrieval_conf < 0.50:
            reasons.append(f"Historical retrieval coverage is critically low (max similarity: {retrieval_conf:.2f})")
            return {
                "decision": "ESCALATE",
                "reasons": reasons,
                "confidence_score": intent_confidence,
                "action_code": "ESC_SPARSE_PRECEDENT"
            }

        # Signal 6: Underspecified customer message requiring clarification
        if len(clean_words) <= self.clarification_min_words:
            reasons.append("Customer inquiry is underspecified or missing context")
            return {
                "decision": "ASK_CLARIFYING_QUESTION",
                "reasons": reasons,
                "confidence_score": intent_confidence,
                "action_code": "CLR_UNDERSPECIFIED"
            }

        # Signal 7: Safe for automated reply
        reasons.append("Sufficient historical precedent verified")
        reasons.append(f"High intent alignment ({evidence_assessment.get('intent_alignment', 1.0)*100:.0f}%)")
        reasons.append(f"Consistent resolution consensus ({evidence_assessment.get('resolution_agreement', 1.0)*100:.0f}%)")
        
        return {
            "decision": "AUTO_HANDLE",
            "reasons": reasons,
            "confidence_score": round((intent_confidence + retrieval_conf) / 2, 4),
            "action_code": "AUTO_GROUNDED_REPLY"
        }

if __name__ == "__main__":
    engine = DecisionEngine()
    # Test case: normal safe inquiry
    res1 = engine.decide(
        "Why is my order 3 days late?",
        "order_delay_tracking",
        0.88,
        {"is_high_risk": False, "explicit_human_requested": False},
        {"evidence_sufficient": True, "evidence_conflicts": False, "retrieval_confidence": 0.82, "intent_alignment": 0.80, "resolution_agreement": 0.90}
    )
    print("Safe Inquiry Decision:", res1)
    
    # Test case: human request
    res2 = engine.decide(
        "I need to speak to a real person",
        "human_agent_complaint",
        0.95,
        {"is_high_risk": True, "explicit_human_requested": True},
        {"evidence_sufficient": False, "evidence_conflicts": False, "retrieval_confidence": 0.60}
    )
    print("Human Request Decision:", res2)
