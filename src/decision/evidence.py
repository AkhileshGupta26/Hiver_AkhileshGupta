"""Evidence validation module for retrieved historical customer support cases.

Assesses semantic similarity distribution, resolution consensus, intent alignment,
and detects contradictory or insufficient historical evidence before generation.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from src.evaluation.leakage import clean_text_for_comparison, compute_jaccard_similarity

class EvidenceValidator:
    """Validates whether retrieved historical cases provide sufficient, coherent evidence."""

    def __init__(
        self,
        strong_similarity_threshold: float = 0.55,
        min_strong_matches: int = 1,
        min_agreement_score: float = 0.25
    ):
        self.strong_threshold = strong_similarity_threshold
        self.min_strong_matches = min_strong_matches
        self.min_agreement = min_agreement_score

    def validate_evidence(
        self,
        predicted_intent: str,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Produce structured evidence validation assessment."""
        if not retrieved_cases:
            return {
                "retrieval_confidence": 0.0,
                "evidence_count": 0,
                "strong_matches": 0,
                "intent_alignment": 0.0,
                "resolution_agreement": 0.0,
                "evidence_conflicts": False,
                "evidence_sufficient": False,
                "validation_notes": ["No historical cases retrieved."]
            }

        sims = [c.get("similarity", 0.0) for c in retrieved_cases]
        retrieval_conf = float(np.max(sims))
        strong_matches = sum(1 for s in sims if s >= self.strong_threshold)

        # Intent alignment: proportion of retrieved examples sharing predicted intent
        matching_intents = sum(1 for c in retrieved_cases if c.get("intent") == predicted_intent)
        intent_alignment = round(matching_intents / len(retrieved_cases), 3)

        # Resolution agreement: compute pairwise agreement among historical agent responses
        agent_responses = [c.get("agent_response", "") for c in retrieved_cases if c.get("agent_response")]
        
        pairwise_agreements = []
        for i in range(len(agent_responses)):
            for j in range(i + 1, len(agent_responses)):
                sim = compute_jaccard_similarity(agent_responses[i], agent_responses[j])
                pairwise_agreements.append(sim)

        mean_agreement = float(np.mean(pairwise_agreements)) if pairwise_agreements else 1.0
        # Normalize agreement score (Jaccard on responses is typically 0.2-0.8)
        norm_agreement = min(1.0, mean_agreement * 2.5) if pairwise_agreements else 1.0
        norm_agreement = round(norm_agreement, 3)

        # Conflict detection
        evidence_conflicts = (strong_matches >= 2 and norm_agreement < 0.25)

        # Evidence sufficiency rule
        evidence_sufficient = (
            strong_matches >= self.min_strong_matches and
            norm_agreement >= self.min_agreement and
            retrieval_conf >= self.strong_threshold and
            intent_alignment >= 0.40 and
            not evidence_conflicts
        )

        validation_notes = []
        if retrieval_conf < self.strong_threshold:
            validation_notes.append(f"Low retrieval confidence ({retrieval_conf:.2f} < {self.strong_threshold})")
        if strong_matches < self.min_strong_matches:
            validation_notes.append(f"Sparse historical precedent (only {strong_matches} strong matches)")
        if intent_alignment < 0.40:
            validation_notes.append(f"Retrieved cases cross intent boundaries (alignment: {intent_alignment:.2f})")
        if evidence_conflicts:
            validation_notes.append("Historical agent resolutions diverge or conflict")

        return {
            "retrieval_confidence": round(retrieval_conf, 4),
            "evidence_count": len(retrieved_cases),
            "strong_matches": strong_matches,
            "intent_alignment": intent_alignment,
            "resolution_agreement": norm_agreement,
            "evidence_conflicts": evidence_conflicts,
            "evidence_sufficient": evidence_sufficient,
            "validation_notes": validation_notes
        }

if __name__ == "__main__":
    validator = EvidenceValidator()
    mock_retrieved = [
        {"intent": "order_delay_tracking", "similarity": 0.82, "agent_response": "We apologize for the delay. Please track via your Orders page or DM us."},
        {"intent": "order_delay_tracking", "similarity": 0.79, "agent_response": "Sorry to hear about the delay. You can check carrier updates on your Orders page."},
        {"intent": "order_delay_tracking", "similarity": 0.71, "agent_response": "Please check tracking on your account page for the latest estimated delivery date."}
    ]
    res = validator.validate_evidence("order_delay_tracking", mock_retrieved)
    print("Validator Output:", res)
