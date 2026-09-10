"""Knowledge gap detection module.

Aggregates repeated customer inquiries where historical evidence is sparse,
conflicting, or low-confidence, providing automated feedback to support operations
and knowledge base teams.
"""

from collections import defaultdict
from typing import Dict, List, Any

class KnowledgeGapTracker:
    """Tracks and aggregates recurring ungrounded customer questions."""

    def __init__(self, gap_frequency_threshold: int = 3):
        self.threshold = gap_frequency_threshold
        # intent -> list of low-evidence queries
        self.gap_clusters = defaultdict(list)

    def record_query(
        self,
        intent: str,
        query: str,
        evidence_assessment: Dict[str, Any]
    ):
        """Record query if evidence was insufficient or weak."""
        is_weak = (
            not evidence_assessment.get("evidence_sufficient", True) or
            evidence_assessment.get("retrieval_confidence", 1.0) < 0.65 or
            evidence_assessment.get("strong_matches", 0) < 2
        )
        if is_weak:
            self.gap_clusters[intent].append({
                "query": query,
                "confidence": evidence_assessment.get("retrieval_confidence", 0.0),
                "strong_matches": evidence_assessment.get("strong_matches", 0)
            })

    def detect_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify intents that exceed the gap threshold."""
        gaps = []
        for intent, queries in self.gap_clusters.items():
            if len(queries) >= self.threshold:
                mean_conf = sum(q["confidence"] for q in queries) / len(queries)
                total_strong = sum(q["strong_matches"] for q in queries)
                
                gaps.append({
                    "intent": intent,
                    "observed_requests": len(queries),
                    "strong_historical_resolutions": total_strong,
                    "evidence_confidence": "LOW" if mean_conf < 0.60 else "MODERATE",
                    "sample_queries": [q["query"] for q in queries[:3]],
                    "recommendation": f"Create standard operating procedure (SOP) and macro response for '{intent}' issues."
                })
        return gaps

    def format_gap_report(self) -> str:
        """Produce formatted markdown report of identified knowledge gaps."""
        gaps = self.detect_knowledge_gaps()
        if not gaps:
            return "No recurring knowledge gaps detected in current session."

        lines = ["# Knowledge Gap Detection Report\n"]
        for g in gaps:
            lines.append(f"### Knowledge Gap Detected: `{g['intent']}`")
            lines.append(f"- **Observed Requests:** {g['observed_requests']}")
            lines.append(f"- **Strong Historical Resolutions:** {g['strong_historical_resolutions']}")
            lines.append(f"- **Evidence Confidence:** {g['evidence_confidence']}")
            lines.append(f"- **Recommendation:** {g['recommendation']}")
            lines.append("- **Sample Inquiries:**")
            for q in g["sample_queries"]:
                lines.append(f"  - *\"{q}\"*")
            lines.append("")
        return "\n".join(lines)
