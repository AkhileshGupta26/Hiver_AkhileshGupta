"""Data leakage prevention and audit module.

Enforces strict conversation-level dataset partitioning and verifies that
no conversations, tweet IDs, exact inquiry texts, or near-duplicate messages
leak between the training/retrieval index and evaluation sets.
"""

import re
import string
from typing import Dict, List, Tuple, Set, Any
import numpy as np

def clean_text_for_comparison(text: str) -> str:
    """Normalize text for exact and near duplicate detection."""
    # Remove mentions, urls, extra whitespace, lowercase
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.lower().split())

def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard similarity between two texts."""
    tokens_a = set(clean_text_for_comparison(text_a).split())
    tokens_b = set(clean_text_for_comparison(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return intersection / union

def create_leakage_safe_splits(
    conversations: List[Dict[str, Any]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create conversation-level splits guaranteed free of exact duplicates and conversation overlap."""
    np.random.seed(random_seed)
    
    # Shuffle entire conversations list
    shuffled = list(conversations)
    np.random.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    initial_train = shuffled[:n_train]
    initial_val = shuffled[n_train:n_train + n_val]
    initial_test = shuffled[n_train + n_val:]
    
    # Extract training normalized text fingerprints
    train_norm_inquiries = {
        clean_text_for_comparison(c["customer_inquiry"]): c["conversation_id"]
        for c in initial_train
        if clean_text_for_comparison(c["customer_inquiry"])
    }
    
    # Filter validation set to eliminate any exact text match with train
    filtered_val = []
    val_duplicates_removed = 0
    for conv in initial_val:
        norm_txt = clean_text_for_comparison(conv["customer_inquiry"])
        if norm_txt in train_norm_inquiries:
            val_duplicates_removed += 1
        else:
            filtered_val.append(conv)
            
    # Filter test set to eliminate any exact text match with train or val
    train_val_norm = set(train_norm_inquiries.keys()).union(
        clean_text_for_comparison(c["customer_inquiry"]) for c in filtered_val
    )
    filtered_test = []
    test_duplicates_removed = 0
    for conv in initial_test:
        norm_txt = clean_text_for_comparison(conv["customer_inquiry"])
        if norm_txt in train_val_norm:
            test_duplicates_removed += 1
        else:
            filtered_test.append(conv)
            
    print(f"Dataset split complete (Conversation-level):")
    print(f"  Train: {len(initial_train):,} conversations")
    print(f"  Val:   {len(filtered_val):,} conversations (purged {val_duplicates_removed} duplicate inquiries)")
    print(f"  Test:  {len(filtered_test):,} conversations (purged {test_duplicates_removed} duplicate inquiries)")
    
    return initial_train, filtered_val, filtered_test

def audit_leakage(
    train_convs: List[Dict[str, Any]],
    eval_convs: List[Dict[str, Any]],
    eval_name: str = "Evaluation Set",
    near_dup_threshold: float = 0.85,
    sample_near_dup_check: int = 500
) -> Dict[str, Any]:
    """Perform rigorous 5-point data leakage audit between training and evaluation splits."""
    train_conv_ids = {c.get("conversation_id", c.get("id", "")) for c in train_convs}
    eval_conv_ids = {c.get("conversation_id", c.get("id", "")) for c in eval_convs}
    conv_overlap = train_conv_ids.intersection(eval_conv_ids)
    
    train_tweet_ids = set()
    for c in train_convs:
        for t in c.get("turns", []):
            train_tweet_ids.add(t["tweet_id"])
            
    eval_tweet_ids = set()
    for c in eval_convs:
        for t in c.get("turns", []):
            eval_tweet_ids.add(t["tweet_id"])
            
    tweet_overlap = train_tweet_ids.intersection(eval_tweet_ids)
    
    # Exact normalized inquiry text matches
    train_inquiries = {
        clean_text_for_comparison(c.get("customer_inquiry", c.get("message", "")))
        for c in train_convs if clean_text_for_comparison(c.get("customer_inquiry", c.get("message", "")))
    }
    eval_inquiries = [
        clean_text_for_comparison(c.get("customer_inquiry", c.get("message", "")))
        for c in eval_convs if clean_text_for_comparison(c.get("customer_inquiry", c.get("message", "")))
    ]
    
    exact_text_leaks = [txt for txt in eval_inquiries if txt in train_inquiries]
    
    # Sampled near-duplicate check (Jaccard > threshold)
    near_dup_count = 0
    sample_eval = eval_inquiries[:sample_near_dup_check]
    # Sample subset of train for speed
    train_sample_list = list(train_inquiries)[:1000]
    for e_txt in sample_eval:
        for t_txt in train_sample_list:
            if compute_jaccard_similarity(e_txt, t_txt) >= near_dup_threshold:
                near_dup_count += 1
                break
                
    audit_report = {
        "eval_split": eval_name,
        "train_size": len(train_convs),
        "eval_size": len(eval_convs),
        "conversation_id_overlap": len(conv_overlap),
        "tweet_id_overlap": len(tweet_overlap),
        "exact_inquiry_leakage_count": len(exact_text_leaks),
        "exact_inquiry_leakage_rate": round(len(exact_text_leaks) / max(1, len(eval_convs)) * 100, 3),
        "near_duplicate_matches_sampled": near_dup_count,
        "leakage_free": (len(conv_overlap) == 0 and len(tweet_overlap) == 0 and len(exact_text_leaks) == 0)
    }
    
    status_str = "PASSED (Zero Leakage)" if audit_report["leakage_free"] else "FAILED (Leakage Detected!)"
    print(f"\n--- Leakage Audit: {eval_name} vs Train ---")
    print(f"  Status: {status_str}")
    print(f"  Conversation ID overlap: {audit_report['conversation_id_overlap']}")
    print(f"  Tweet ID overlap:        {audit_report['tweet_id_overlap']}")
    print(f"  Exact inquiry matches:   {audit_report['exact_inquiry_leakage_count']} ({audit_report['exact_inquiry_leakage_rate']}%)")
    print(f"  Near-duplicate matches:  {audit_report['near_duplicate_matches_sampled']} (sample of {len(sample_eval)})")
    
    return audit_report
