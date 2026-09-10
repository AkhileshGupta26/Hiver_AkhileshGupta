"""Abstention and coverage tradeoff analysis module.

Evaluates how varying confidence and evidence thresholds impacts automation coverage,
accuracy, response quality, and critical unsafe auto-handle rates.
"""

from typing import List, Dict, Any
import numpy as np

def run_abstention_analysis(
    evaluation_records: List[Dict[str, Any]],
    thresholds: List[float] = [0.50, 0.60, 0.70, 0.80, 0.90]
) -> List[Dict[str, Any]]:
    """Generate empirical abstention curve across confidence thresholds."""
    print("\n--- Running Abstention & Coverage Tradeoff Analysis ---")
    curve_data = []
    total_inquiries = len(evaluation_records)

    for tau in thresholds:
        auto_handled_items = []
        escalated_items = []

        for r in evaluation_records:
            conf = r.get("intent_confidence", 0.0)
            is_risk = r.get("is_high_risk", False)
            ev_sufficient = r.get("evidence_sufficient", True)
            
            # Policy at threshold tau
            if conf >= tau and not is_risk and ev_sufficient:
                auto_handled_items.append(r)
            else:
                escalated_items.append(r)

        coverage_pct = (len(auto_handled_items) / total_inquiries) * 100

        if auto_handled_items:
            # Accuracy on auto-handled subset
            correct_intents = sum(1 for item in auto_handled_items if item["predicted_intent"] == item["expected_intent"])
            intent_acc = (correct_intents / len(auto_handled_items)) * 100

            # Judge quality score on auto-handled subset
            judge_scores = [item.get("judge_score", 8.0) for item in auto_handled_items]
            mean_quality = float(np.mean(judge_scores))

            # Unsafe auto-handle rate: Auto-handled but should have escalated
            unsafe_auto_count = sum(1 for item in auto_handled_items if item.get("should_escalate", False))
            unsafe_rate = (unsafe_auto_count / len(auto_handled_items)) * 100
        else:
            intent_acc = 100.0
            mean_quality = 10.0
            unsafe_rate = 0.0

        point = {
            "threshold": tau,
            "coverage_percent": round(coverage_pct, 2),
            "auto_handled_count": len(auto_handled_items),
            "intent_accuracy": round(intent_acc, 2),
            "mean_response_quality": round(mean_quality, 2),
            "unsafe_auto_handle_rate": round(unsafe_rate, 2)
        }
        curve_data.append(point)
        print(f"  Threshold {tau:.2f} | Coverage: {coverage_pct:5.1f}% | Acc: {intent_acc:5.1f}% | Quality: {mean_quality:4.1f}/10 | Unsafe Auto-Handle: {unsafe_rate:4.1f}%")

    return curve_data
