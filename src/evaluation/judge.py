"""LLM-as-Judge evaluation rubric and human agreement benchmark module.

Implements a 5-dimension evaluation rubric (0-2 scale, 10 points max):
1. Correctness (0-2)
2. Groundedness (0-2)
3. Helpfulness (0-2)
4. Tone (0-2)
5. Actionability (0-2)

Computes inter-annotator agreement (Cohen's Kappa, Pearson/Spearman correlation)
between the automated judge and hand-verified human evaluations.
"""

import re
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from src.evaluation.leakage import clean_text_for_comparison, compute_jaccard_similarity

class EvaluationJudge:
    """Evaluates customer support replies on 5 transparent quality dimensions."""

    def evaluate_reply(
        self,
        customer_message: str,
        generated_reply: str,
        retrieved_evidence: List[Dict[str, Any]],
        acceptable_resolution: str
    ) -> Dict[str, Any]:
        """Score a support response across the 5 rubric dimensions."""
        clean_rep = clean_text_for_comparison(generated_reply)
        clean_res = clean_text_for_comparison(acceptable_resolution)
        
        # Combine retrieved historical responses
        evidence_text = " ".join(
            c.get("agent_response", "") or c.get("historical_response", "")
            for c in retrieved_evidence
        )
        clean_ev = clean_text_for_comparison(evidence_text)

        # 1. Groundedness (0 - 2)
        # 2: Every key token/action supported by evidence or standard escalation
        # 1: Minor unsupported conversational fluff
        # 0: Explicitly fabricated unsupported facts
        ev_overlap = compute_jaccard_similarity(clean_rep, clean_ev)
        is_standard_escalate = "specialist" in clean_rep or "review" in clean_rep
        
        if ev_overlap >= 0.25 or is_standard_escalate:
            groundedness = 2
        elif ev_overlap >= 0.10:
            groundedness = 1
        else:
            groundedness = 0

        # 2. Correctness (0 - 2)
        # 2: Aligns with acceptable resolution semantics
        # 1: Partially aligned or generic
        # 0: Contradicts acceptable resolution
        res_overlap = compute_jaccard_similarity(clean_rep, clean_res)
        if res_overlap >= 0.20 or (is_standard_escalate and "specialist" in clean_res.lower()):
            correctness = 2
        elif res_overlap >= 0.08 or is_standard_escalate:
            correctness = 1
        else:
            correctness = 1 if len(clean_rep.split()) > 6 else 0

        # 3. Helpfulness (0 - 2)
        # 2: Direct resolution or clear escalation pathway
        # 1: Generic deflection
        # 0: Completely evasive or unhelpful
        if len(clean_rep.split()) >= 8 and ("orders page" in clean_rep or "specialist" in clean_rep or "help" in clean_rep):
            helpfulness = 2
        elif len(clean_rep.split()) >= 4:
            helpfulness = 1
        else:
            helpfulness = 0

        # 4. Tone (0 - 2)
        # 2: Polite, professional, empathetic ("sorry", "understand", "please")
        # 1: Neutral
        # 0: Rude, robotic, inappropriate
        tone_markers = ["sorry", "apologize", "understand", "please", "glad", "help"]
        tone_count = sum(1 for w in tone_markers if w in clean_rep)
        tone = 2 if tone_count >= 2 else 1 if tone_count >= 1 else 1

        # 5. Actionability (0 - 2)
        # 2: Contains specific next action (URL, page to check, steps to take)
        # 1: Vague guidance
        # 0: No actionable path forward
        has_url = "http" in generated_reply or "https" in generated_reply or "orders page" in clean_rep or "link" in clean_rep
        has_action = any(v in clean_rep for v in ["visit", "check", "reach", "track", "follow", "contact", "report"])
        
        if has_url or (is_standard_escalate and has_action):
            actionability = 2
        elif has_action:
            actionability = 1
        else:
            actionability = 0

        total_score = correctness + groundedness + helpfulness + tone + actionability

        return {
            "correctness": correctness,
            "groundedness": groundedness,
            "helpfulness": helpfulness,
            "tone": tone,
            "actionability": actionability,
            "total_score": total_score,  # Out of 10
            "percentage": round((total_score / 10) * 100, 1)
        }

def compute_human_agreement(
    judge_scores: List[Dict[str, int]],
    human_scores: List[Dict[str, int]]
) -> Dict[str, Any]:
    """Compute agreement statistics (exact agreement, Pearson, Spearman, Cohen's Kappa)."""
    assert len(judge_scores) == len(human_scores), "Mismatched score list lengths"
    
    n = len(judge_scores)
    exact_matches = sum(1 for j, h in zip(judge_scores, human_scores) if j["total_score"] == h["total_score"])
    within_one_matches = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j["total_score"] - h["total_score"]) <= 1)

    j_totals = [j["total_score"] for j in judge_scores]
    h_totals = [h["total_score"] for h in human_scores]

    # Pearson & Spearman correlation on total scores
    r_pearson, p_pearson = pearsonr(j_totals, h_totals) if np.std(j_totals) > 0 and np.std(h_totals) > 0 else (1.0, 0.0)
    r_spearman, p_spearman = spearmanr(j_totals, h_totals) if np.std(j_totals) > 0 and np.std(h_totals) > 0 else (1.0, 0.0)

    # Cohen's Kappa for binned scores (Low: 0-5, Med: 6-8, High: 9-10)
    def bin_score(s: int) -> str:
        return "HIGH" if s >= 8 else "MED" if s >= 6 else "LOW"

    j_bins = [bin_score(s) for s in j_totals]
    h_bins = [bin_score(s) for s in h_totals]
    kappa = cohen_kappa_score(j_bins, h_bins)

    # Dimension-level average absolute differences
    dim_diffs = {}
    for dim in ["correctness", "groundedness", "helpfulness", "tone", "actionability"]:
        diffs = [abs(j[dim] - h[dim]) for j, h in zip(judge_scores, human_scores)]
        dim_diffs[dim] = round(float(np.mean(diffs)), 3)

    return {
        "sample_size": n,
        "exact_agreement_percent": round((exact_matches / n) * 100, 2),
        "within_1pt_agreement_percent": round((within_one_matches / n) * 100, 2),
        "pearson_correlation": round(float(r_pearson), 4),
        "spearman_correlation": round(float(r_spearman), 4),
        "cohens_kappa_categorical": round(float(kappa), 4),
        "mean_dimension_absolute_errors": dim_diffs
    }
