"""Dual baseline models for intent classification and response retrieval.

Baseline 1: Trivial Majority-Class Classifier and Majority Response.
Baseline 2: Non-LLM TF-IDF + Logistic Regression Classifier and TF-IDF Cosine Retrieval.
"""

import os
import json
import pickle
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from src.evaluation.leakage import clean_text_for_comparison

class MajorityBaseline:
    """Baseline 1: Trivial majority intent classifier and majority response."""
    
    def __init__(self):
        self.majority_intent: Optional[str] = None
        self.majority_response: Optional[str] = None
        self.majority_frequency: float = 0.0
        
    def fit(self, train_data: List[Dict[str, Any]]):
        intents = [d["intent"] for d in train_data]
        intent_counter = Counter(intents)
        self.majority_intent, count = intent_counter.most_common(1)[0]
        self.majority_frequency = count / len(train_data)
        
        # Most frequent or representative agent response for majority class
        majority_responses = [
            d["agent_resolution"] for d in train_data
            if d["intent"] == self.majority_intent and d.get("agent_resolution")
        ]
        resp_counter = Counter(majority_responses)
        self.majority_response = resp_counter.most_common(1)[0][0]
        print(f"MajorityBaseline fitted: Intent='{self.majority_intent}' (Freq: {self.majority_frequency:.3f})")
        
    def predict_intent(self, text: str) -> Tuple[str, float]:
        return self.majority_intent, self.majority_frequency
        
    def generate_response(self, text: str) -> str:
        return self.majority_response

class TfidfLogisticBaseline:
    """Baseline 2: TF-IDF + Logistic Regression Intent Classifier and TF-IDF Cosine Retriever."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True
        )
        self.classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        self.classes_: Optional[np.ndarray] = None
        
        # Retrieval store
        self.train_inquiries: List[str] = []
        self.train_responses: List[str] = []
        self.train_conv_ids: List[str] = []
        self.train_intents: List[str] = []
        self.train_tfidf_matrix = None
        
    def fit(self, train_data: List[Dict[str, Any]]):
        print(f"Fitting TfidfLogisticBaseline on {len(train_data):,} training samples...")
        raw_inquiries = [d["customer_inquiry"] for d in train_data]
        cleaned_inquiries = [clean_text_for_comparison(t) for t in raw_inquiries]
        labels = [d["intent"] for d in train_data]
        
        X_train = self.vectorizer.fit_transform(cleaned_inquiries)
        self.classifier.fit(X_train, labels)
        self.classes_ = self.classifier.classes_
        
        # Store for TF-IDF cosine retrieval
        self.train_inquiries = raw_inquiries
        self.train_responses = [d.get("agent_resolution", "") for d in train_data]
        self.train_conv_ids = [d["conversation_id"] for d in train_data]
        self.train_intents = labels
        self.train_tfidf_matrix = X_train
        print("TfidfLogisticBaseline training complete.")
        
    def predict_intent(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        cleaned = clean_text_for_comparison(text)
        X_vec = self.vectorizer.transform([cleaned])
        probs = self.classifier.predict_proba(X_vec)[0]
        pred_idx = np.argmax(probs)
        pred_label = self.classes_[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(self.classes_)}
        return pred_label, confidence, prob_dict
        
    def retrieve_response(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        cleaned = clean_text_for_comparison(text)
        q_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(q_vec, self.train_tfidf_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "conversation_id": self.train_conv_ids[idx],
                "inquiry": self.train_inquiries[idx],
                "agent_response": self.train_responses[idx],
                "intent": self.train_intents[idx],
                "similarity": float(sims[idx])
            })
        return results
        
    def generate_response(self, text: str) -> str:
        matches = self.retrieve_response(text, top_k=1)
        if matches and matches[0]["agent_response"]:
            return matches[0]["agent_response"]
        return "Please contact Amazon customer support with your order details."

def evaluate_classifier(y_true: List[str], y_pred: List[str], model_name: str) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    unique_labels = sorted(list(set(y_true)))
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )
    
    per_intent = {
        label: {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i])
        }
        for i, label in enumerate(unique_labels)
    }
    
    report = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "macro_precision": round(p_macro, 4),
        "macro_recall": round(r_macro, 4),
        "weighted_f1": round(f1_wt, 4),
        "per_intent": per_intent
    }
    return report

def run_baseline_evaluation(
    train_file: str = "data/train.jsonl",
    test_file: str = "data/test.jsonl",
    output_json: str = "evaluation/results/baseline_metrics.json"
):
    print("Loading train and test splits...")
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    print(f"Train samples: {len(train_data):,} | Test samples: {len(test_data):,}")
    y_test = [d["intent"] for d in test_data]
    test_inquiries = [d["customer_inquiry"] for d in test_data]

    # 1. Baseline 1: Majority Class
    print("\n--- Evaluating Baseline 1: Majority Class ---")
    b1 = MajorityBaseline()
    b1.fit(train_data)
    y_pred_b1 = [b1.predict_intent(t)[0] for t in test_inquiries]
    b1_report = evaluate_classifier(y_test, y_pred_b1, "Baseline 1: Majority Class")

    # 2. Baseline 2: TF-IDF + Logistic Regression
    print("\n--- Evaluating Baseline 2: TF-IDF + Logistic Regression ---")
    b2 = TfidfLogisticBaseline()
    b2.fit(train_data)
    
    y_pred_b2 = []
    b2_sims = []
    for t in test_inquiries:
        pred_label, conf, _ = b2.predict_intent(t)
        y_pred_b2.append(pred_label)
        retrieved = b2.retrieve_response(t, top_k=1)
        b2_sims.append(retrieved[0]["similarity"])
        
    b2_report = evaluate_classifier(y_test, y_pred_b2, "Baseline 2: TF-IDF + Logistic Regression")
    b2_report["mean_retrieval_similarity"] = round(float(np.mean(b2_sims)), 4)

    # Save results
    results = {
        "baseline_1_majority": b1_report,
        "baseline_2_tfidf_logistic": b2_report
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved baseline evaluation results to: {output_json}")
    print("\n=== BASELINE COMPARISON SUMMARY ===")
    print(f"{'Model':<35} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12}")
    print("-" * 75)
    print(f"{b1_report['model']:<35} | {b1_report['accuracy']*100:<9.2f}% | {b1_report['macro_f1']:<10.4f} | {b1_report['weighted_f1']:<12.4f}")
    print(f"{b2_report['model']:<35} | {b2_report['accuracy']*100:<9.2f}% | {b2_report['macro_f1']:<10.4f} | {b2_report['weighted_f1']:<12.4f}")

if __name__ == "__main__":
    run_baseline_evaluation()
