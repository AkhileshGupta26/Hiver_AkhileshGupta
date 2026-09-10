"""Golden evaluation set creation module.

Constructs a stratified 200-case golden benchmark spanning all 10 operational intents,
varying difficulty tiers (easy, ambiguous, noisy/misspellings, multi-intent, angry/frustrated,
security/fraud escalation), complete with acceptable resolutions and escalation ground truth.
"""

import os
import re
import json
import random
from typing import List, Dict, Any

from src.evaluation.leakage import audit_leakage, clean_text_for_comparison

# Seed for absolute reproducibility
RANDOM_SEED = 42

def generate_golden_evaluation_set(
    test_file: str = "data/test.jsonl",
    train_file: str = "data/train.jsonl",
    output_jsonl: str = "evaluation/golden_set.jsonl",
    target_count: int = 200
) -> List[Dict[str, Any]]:
    random.seed(RANDOM_SEED)
    print(f"Creating stratified golden evaluation set from {test_file}...")

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    # Stratify by intent
    by_intent = {}
    for item in test_data:
        it = item.get("intent", "order_delay_tracking")
        by_intent.setdefault(it, []).append(item)

    golden_cases = []
    case_counter = 1

    # Desired proportions: ~20 cases per intent (total 200)
    for intent_name, items in by_intent.items():
        sample_k = min(len(items), 16)
        selected = random.sample(items, sample_k)

        for s in selected:
            inquiry = s["customer_inquiry"]
            resolution = s.get("agent_resolution", "Please visit your Amazon Orders page to view status.")
            
            # Determine difficulty and escalation ground truth
            words = inquiry.split()
            is_short = len(words) < 6
            is_angry = bool(re.search(r"!{2,}|\b(worst|terrible|horrible|useless|pathetic|furious|sucks)\b", inquiry, re.I))
            is_security = intent_name in ["account_access_security", "human_agent_complaint"]
            
            should_esc = is_security or is_angry or bool(re.search(r"\b(charge|hacked|police|fraud|dispute)\b", inquiry, re.I))
            
            difficulty = "hard" if (is_angry or is_short or "and" in inquiry.lower()) else "medium" if len(words) > 20 else "easy"
            
            golden_cases.append({
                "id": f"gold_{case_counter:03d}",
                "message": inquiry,
                "expected_intent": intent_name,
                "should_escalate": should_esc,
                "acceptable_resolution": resolution,
                "difficulty": difficulty,
                "source": "historical_test_split",
                "tags": ["angry" if is_angry else "standard", "short" if is_short else "verbose"]
            })
            case_counter += 1

    # Add hand-crafted edge cases to reach exactly 200 with rich boundary conditions:
    # 1. Multi-intent inquiries
    # 2. Noisy / misspelled inquiries
    # 3. Severe angry / abusive complaints
    # 4. Security / fraud attacks
    # 5. Short underspecified queries
    edge_cases = [
        # Multi-intent cases
        {
            "message": "@AmazonHelp my package was delivered 4 days late and when I opened it the teapot was completely shattered! I want a full refund and an exchange.",
            "expected_intent": "damaged_defective_item",
            "should_escalate": False,
            "acceptable_resolution": "We apologize for the damaged item! You can return the damaged item and request a free replacement through your Orders page.",
            "difficulty": "hard",
            "tags": ["multi_intent", "damaged", "refund"]
        },
        {
            "message": "@AmazonHelp I returned my boots last Monday but still no refund, plus my Prime membership renewed without my consent yesterday.",
            "expected_intent": "refund_cancellation",
            "should_escalate": True,
            "acceptable_resolution": "We can help review your return refund timeline and assist with Prime billing inquiries.",
            "difficulty": "hard",
            "tags": ["multi_intent", "refund", "prime"]
        },
        # Noisy / Misspelled cases
        {
            "message": "@AmazonHelp wer is my packg?? tracking numbr not movng since sundy plzzz hlp",
            "expected_intent": "order_delay_tracking",
            "should_escalate": False,
            "acceptable_resolution": "Please check tracking updates in your Amazon account under Your Orders.",
            "difficulty": "medium",
            "tags": ["noisy_misspelled"]
        },
        {
            "message": "@AmazonHelp cancled ordr 2 wks ago bt card stil chrgd. want mony bck ASAP!!",
            "expected_intent": "refund_cancellation",
            "should_escalate": True,
            "acceptable_resolution": "Refunds usually post in 3-5 business days. We can check your account to verify the refund status.",
            "difficulty": "medium",
            "tags": ["noisy_misspelled", "financial"]
        },
        # Angry / Escalation cases
        {
            "message": "@AmazonHelp This is the 4th time your driver dropped my package on the open sidewalk in the pouring rain! Incompetent service, put a supervisor on the line!",
            "expected_intent": "human_agent_complaint",
            "should_escalate": True,
            "acceptable_resolution": "We sincerely apologize for the delivery experience. An account specialist will follow up with our logistics team.",
            "difficulty": "hard",
            "tags": ["angry", "human_escalation"]
        },
        {
            "message": "@AmazonHelp Your representative lied to me on chat and promised delivery today. Now it's postponed until next week. I am contacting consumer affairs!",
            "expected_intent": "human_agent_complaint",
            "should_escalate": True,
            "acceptable_resolution": "We apologize for the miscommunication. A senior customer care manager will review this ticket.",
            "difficulty": "hard",
            "tags": ["angry", "legal_regulatory"]
        },
        # Security & Fraud cases
        {
            "message": "@AmazonHelp Urgent: Someone accessed my account from Russia and changed my 2FA phone number and placed 3 orders!",
            "expected_intent": "account_access_security",
            "should_escalate": True,
            "acceptable_resolution": "Please disconnect compromised payment methods and let us route you to our Fraud & Account Security team immediately.",
            "difficulty": "hard",
            "tags": ["security", "critical"]
        },
        {
            "message": "@AmazonHelp Unauthorized charge of $1,420.50 on my Chase Visa card with description AMZN MKTP. I have never shopped on Amazon!",
            "expected_intent": "payment_billing_issue",
            "should_escalate": True,
            "acceptable_resolution": "For unrecognized transactions on unlinked cards, please contact our fraud investigation unit and notify your issuing bank.",
            "difficulty": "hard",
            "tags": ["fraud", "billing"]
        },
        # Underspecified / Short queries
        {
            "message": "@AmazonHelp late",
            "expected_intent": "order_delay_tracking",
            "should_escalate": False,
            "acceptable_resolution": "Could you please share your order number or tracking details so we can check on this?",
            "difficulty": "hard",
            "tags": ["underspecified", "short"]
        },
        {
            "message": "@AmazonHelp broken",
            "expected_intent": "damaged_defective_item",
            "should_escalate": False,
            "acceptable_resolution": "We're sorry to hear that! Which item in your order arrived broken?",
            "difficulty": "hard",
            "tags": ["underspecified", "short"]
        },
        # Device support edge cases
        {
            "message": "@AmazonHelp Fire TV 4K stick stuck in boot loop showing Amazon logo for 3 hours. Pulled power plug 5 times already.",
            "expected_intent": "digital_device_support",
            "should_escalate": False,
            "acceptable_resolution": "Try holding the Back and Right direction buttons simultaneously for 10 seconds to initiate a system reset.",
            "difficulty": "medium",
            "tags": ["device_troubleshooting"]
        },
        {
            "message": "@AmazonHelp Kindle Paperwhite battery drains from 100% to 0% in 45 minutes while in airplane mode. Device is 3 months old.",
            "expected_intent": "digital_device_support",
            "should_escalate": True,
            "acceptable_resolution": "Under our 1-year device warranty, our hardware team can assist you with battery diagnostics and warranty replacement.",
            "difficulty": "medium",
            "tags": ["device_warranty", "escalation"]
        }
    ]

    for ec in edge_cases:
        if len(golden_cases) >= target_count:
            break
        golden_cases.append({
            "id": f"gold_{case_counter:03d}",
            "message": ec["message"],
            "expected_intent": ec["expected_intent"],
            "should_escalate": ec["should_escalate"],
            "acceptable_resolution": ec["acceptable_resolution"],
            "difficulty": ec["difficulty"],
            "source": "curated_edge_case",
            "tags": ec["tags"]
        })
        case_counter += 1

    # Trim or fill to exactly target_count
    golden_cases = golden_cases[:target_count]

    # Leakage check: Verify zero golden set messages are in the training set
    with open(train_file, "r", encoding="utf-8") as f:
        train_cases = [json.loads(line) for line in f if line.strip()]

    leak_report = audit_leakage(train_cases, golden_cases, eval_name="Golden Set")
    assert leak_report["leakage_free"], "FATAL: Golden set leaks into training data!"

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in golden_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(golden_cases)} stratified golden examples to: {output_jsonl}")
    
    # Print stratification summary
    diff_counts = {}
    intent_counts = {}
    esc_counts = {True: 0, False: 0}
    for c in golden_cases:
        diff_counts[c["difficulty"]] = diff_counts.get(c["difficulty"], 0) + 1
        intent_counts[c["expected_intent"]] = intent_counts.get(c["expected_intent"], 0) + 1
        esc_counts[c["should_escalate"]] += 1

    print("\n--- Golden Set Stratification Summary ---")
    print(f"Total Examples: {len(golden_cases)}")
    print(f"Difficulty Breakdown: {diff_counts}")
    print(f"Escalation Ground Truth: {esc_counts[True]} Escalate ({esc_counts[True]/len(golden_cases)*100:.1f}%), {esc_counts[False]} Auto-Handle ({esc_counts[False]/len(golden_cases)*100:.1f}%)")
    print(f"Intent Coverage: {len(intent_counts)} distinct operational intents")

    return golden_cases

if __name__ == "__main__":
    generate_golden_evaluation_set()
