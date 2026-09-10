"""End-to-end evidence-first support agent pipeline.

Orchestrates message parsing, intent classification, risk assessment,
dense FAISS retrieval, evidence validation, multi-signal triage decisions,
grounded response generation, and structured observability logging.
"""

import os
import json
import time
import uuid
from typing import Dict, Any, Optional, List

from src.retrieval.retriever import DenseRetriever
from src.decision.risk import RiskDetector
from src.decision.evidence import EvidenceValidator
from src.decision.escalation import DecisionEngine
from src.decision.knowledge_gaps import KnowledgeGapTracker
from src.generation.agent import GroundedGenerator
from src.intents.baselines import TfidfLogisticBaseline
from src.evaluation.leakage import clean_text_for_comparison

class SupportAgentPipeline:
    """Production-grade evidence-first support agent pipeline."""

    def __init__(
        self,
        index_dir: str = "models/faiss_index",
        train_path: str = "data/train.jsonl"
    ):
        print("Initializing SupportAgentPipeline...")
        self.risk_detector = RiskDetector()
        self.evidence_validator = EvidenceValidator()
        self.decision_engine = DecisionEngine()
        self.gap_tracker = KnowledgeGapTracker()
        self.generator = GroundedGenerator()
        
        # Initialize Dense Retriever
        self.retriever = DenseRetriever(index_dir=index_dir)
        try:
            self.retriever.load()
        except Exception:
            print("No cached FAISS index found. Building from training data...")
            with open(train_path, "r", encoding="utf-8") as f:
                train_data = [json.loads(line) for line in f if line.strip()]
            self.retriever.build_index(train_data)
            self.retriever.save()

        # Initialize Intent Classifier
        self.intent_classifier = TfidfLogisticBaseline()
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f if line.strip()]
        self.intent_classifier.fit(train_data)
        print("SupportAgentPipeline ready.")

    def process_message(self, customer_message: str, top_k_retrieval: int = 5) -> Dict[str, Any]:
        """Process incoming customer inquiry through full evidence-first architecture."""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # 1. Intent Classification
        pred_intent, intent_conf, prob_dist = self.intent_classifier.predict_intent(customer_message)

        # 2. Risk & Safety Detection
        risk_data = self.risk_detector.assess_risk(customer_message)

        # 3. Dense Historical Case Retrieval
        retrieved_cases = self.retriever.retrieve(customer_message, top_k=top_k_retrieval)

        # 4. Evidence Validation
        evidence_data = self.evidence_validator.validate_evidence(pred_intent, retrieved_cases)

        # 5. Track Knowledge Gaps
        self.gap_tracker.record_query(pred_intent, customer_message, evidence_data)

        # 6. Multi-Signal Decision Engine
        decision_data = self.decision_engine.decide(
            customer_message,
            pred_intent,
            intent_conf,
            risk_data,
            evidence_data
        )

        # 7. Grounded Response Generation
        gen_result = self.generator.synthesize_response(
            customer_message,
            pred_intent,
            decision_data,
            retrieved_cases
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # 8. Structured Observability Log (Requirement 23)
        structured_log = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "latency_ms": elapsed_ms,
            "customer_inquiry": customer_message,
            "predicted_intent": pred_intent,
            "intent_confidence": round(intent_conf, 4),
            "risk_tier": risk_data["risk_tier"],
            "risk_triggers": risk_data["triggered_categories"],
            "retrieval_confidence": evidence_data["retrieval_confidence"],
            "retrieval_top_k": len(retrieved_cases),
            "strong_evidence_matches": evidence_data["strong_matches"],
            "resolution_agreement": evidence_data["resolution_agreement"],
            "evidence_sufficient": evidence_data["evidence_sufficient"],
            "decision": decision_data["decision"],
            "action_code": decision_data.get("action_code", "UNKNOWN"),
            "decision_reasons": decision_data["reasons"],
            "generated_response": gen_result["response"],
            "response_grounded": gen_result["is_grounded"],
            "retrieved_evidence": [
                {
                    "conversation_id": c["conversation_id"],
                    "similarity": c["similarity"],
                    "intent": c["intent"],
                    "historical_response": c["agent_response"]
                }
                for c in retrieved_cases[:3]
            ]
        }

        return structured_log

if __name__ == "__main__":
    pipeline = SupportAgentPipeline()
    sample_queries = [
        "Where is my package? The courier says it was delayed 3 days ago.",
        "I demand to speak with a human agent right now!",
        "Someone hacked into my Amazon account and changed my security settings.",
        "How do I cancel my Amazon Prime free trial before it charges me?"
    ]
    for q in sample_queries:
        print("\n" + "=" * 60)
        print(f"Customer: {q}")
        res = pipeline.process_message(q)
        print(f"Intent:   {res['predicted_intent']} (Conf: {res['intent_confidence']:.2f})")
        print(f"Risk:     {res['risk_tier']}")
        print(f"Decision: {res['decision']} -> Reasons: {res['decision_reasons']}")
        print(f"Response: {res['generated_response']}")
