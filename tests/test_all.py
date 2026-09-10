"""Automated test suite for the Hiver Support Agent pipeline."""

import pytest
from src.evaluation.leakage import clean_text_for_comparison, compute_jaccard_similarity, create_leakage_safe_splits, audit_leakage
from src.decision.risk import RiskDetector
from src.decision.evidence import EvidenceValidator
from src.decision.escalation import DecisionEngine
from src.generation.agent import GroundedGenerator

def test_clean_text():
    raw = "@AmazonHelp where is my package?? https://t.co/xyz123  Order #123"
    cleaned = clean_text_for_comparison(raw)
    assert "@" not in cleaned
    assert "https" not in cleaned
    assert "package" in cleaned

def test_jaccard_similarity():
    s1 = "where is my order package"
    s2 = "where is my delivery package"
    sim = compute_jaccard_similarity(s1, s2)
    assert 0.4 < sim < 1.0

def test_leakage_audit_clean():
    train_mock = [
        {"conversation_id": "c1", "customer_inquiry": "Where is my book?", "turns": [{"tweet_id": "101"}]},
        {"conversation_id": "c2", "customer_inquiry": "How do I return shoes?", "turns": [{"tweet_id": "102"}]}
    ]
    test_mock = [
        {"conversation_id": "c3", "customer_inquiry": "My laptop is broken", "turns": [{"tweet_id": "103"}]}
    ]
    audit = audit_leakage(train_mock, test_mock, eval_name="Test")
    assert audit["leakage_free"] is True
    assert audit["conversation_id_overlap"] == 0
    assert audit["tweet_id_overlap"] == 0
    assert audit["exact_inquiry_leakage_count"] == 0

def test_risk_detector_security():
    detector = RiskDetector()
    res = detector.assess_risk("Someone hacked into my account and changed my password!")
    assert res["is_high_risk"] is True
    assert res["risk_tier"] == "CRITICAL"
    assert "SECURITY_ACCOUNT_COMPROMISE" in res["triggered_categories"]

def test_risk_detector_human_request():
    detector = RiskDetector()
    res = detector.assess_risk("I want to speak to a real person right now.")
    assert res["explicit_human_requested"] is True
    assert res["is_high_risk"] is True

def test_risk_detector_fraud():
    detector = RiskDetector()
    res = detector.assess_risk("I want a refund for an unauthorized $5,000 charge on my credit card.")
    assert res["is_high_risk"] is True
    assert "FINANCIAL_FRAUD_DISPUTE" in res["triggered_categories"]

def test_evidence_validator():
    validator = EvidenceValidator()
    cases = [
        {"similarity": 0.85, "intent": "order_delay_tracking", "agent_response": "Please check your orders page."},
        {"similarity": 0.81, "intent": "order_delay_tracking", "agent_response": "You can check tracking on your account page."}
    ]
    val = validator.validate_evidence("order_delay_tracking", cases)
    assert val["retrieval_confidence"] == 0.85
    assert val["strong_matches"] >= 1
    assert val["intent_alignment"] == 1.0
    assert val["evidence_sufficient"] is True

def test_decision_engine_escalate_on_risk():
    engine = DecisionEngine()
    dec = engine.decide(
        "Someone hacked my account",
        "account_access_security",
        0.95,
        {"is_high_risk": True, "triggered_categories": ["SECURITY_ACCOUNT_COMPROMISE"]},
        {"evidence_sufficient": True, "evidence_conflicts": False}
    )
    assert dec["decision"] == "ESCALATE"
    assert any("Security or policy risk" in r for r in dec["reasons"])

def test_decision_engine_clarify_on_short():
    engine = DecisionEngine()
    dec = engine.decide(
        "late",
        "order_delay_tracking",
        0.40,
        {"is_high_risk": False, "triggered_categories": []},
        {"evidence_sufficient": True, "evidence_conflicts": False}
    )
    assert dec["decision"] == "ASK_CLARIFYING_QUESTION"

def test_grounded_generator_anti_hallucination():
    generator = GroundedGenerator()
    dec = {"decision": "AUTO_HANDLE", "reasons": ["Sufficient evidence"], "confidence_score": 0.90}
    evidence = [{"conversation_id": "c1", "agent_response": "We apologize for the delay. Track at https://amzn.to/help ^SG"}]
    
    reply = generator.synthesize_response("Where is my package?", "order_delay_tracking", dec, evidence)
    assert reply["is_grounded"] is True
    assert reply["safety_checks_passed"] is True
    assert "^SG" not in reply["response"]
