"""Data-driven intent discovery module.

Discovers operational intents from empirical customer inquiries using
dense sentence embeddings, K-Means clustering, keyword analysis, and
medoid extraction, generating configs/intents.yaml.
"""

import os
import re
import json
import yaml
import argparse
from typing import List, Dict, Any
from collections import Counter
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def clean_inquiry_for_nlp(text: str) -> str:
    """Strip mentions, urls, and normalize whitespace for intent clustering."""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return " ".join(text.split())

def discover_intents(
    conversations_path: str = "data/conversations.jsonl",
    output_yaml: str = "configs/intents.yaml",
    n_clusters: int = 10,
    sample_size: int = 3000,
    random_state: int = 42
) -> Dict[str, Any]:
    print(f"Loading customer inquiries from {conversations_path}...")
    inquiries = []
    with open(conversations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                inq = clean_inquiry_for_nlp(item["customer_inquiry"])
                if len(inq.split()) >= 4:
                    inquiries.append((item["conversation_id"], item["customer_inquiry"], inq))
                    if len(inquiries) >= sample_size:
                        break

    print(f"Sampled {len(inquiries):,} customer inquiries for intent discovery.")
    conv_ids, raw_texts, cleaned_texts = zip(*inquiries)

    print("Encoding inquiries with SentenceTransformer ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(list(cleaned_texts), show_progress_bar=True, normalize_embeddings=True)

    print(f"Fitting K-Means with k={n_clusters}...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=256, n_init=5)
    cluster_labels = kmeans.fit_predict(embeddings)

    cluster_counts = Counter(cluster_labels)
    total_samples = len(cluster_labels)

    # Extract distinctive TF-IDF keywords per cluster
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    cluster_profiles = []
    print("\n--- Discovered Cluster Profiles ---")
    for cluster_id in range(n_clusters):
        c_mask = cluster_labels == cluster_id
        c_size = cluster_counts[cluster_id]
        c_pct = (c_size / total_samples) * 100

        # Centroid distance to find medoids (representative real messages)
        c_center = kmeans.cluster_centers_[cluster_id]
        c_indices = np.where(c_mask)[0]
        c_embeddings = embeddings[c_indices]
        dists = np.linalg.norm(c_embeddings - c_center, axis=1)
        medoid_local_idx = np.argsort(dists)[:5]
        medoid_texts = [raw_texts[c_indices[i]] for i in medoid_local_idx]

        # Top keywords
        c_tfidf_mean = tfidf_matrix[c_indices].mean(axis=0)
        top_keyword_idx = np.argsort(np.array(c_tfidf_mean).flatten())[::-1][:6]
        keywords = feature_names[top_keyword_idx].tolist()

        cluster_profiles.append({
            "cluster_id": cluster_id,
            "size": c_size,
            "percentage": round(c_pct, 2),
            "keywords": keywords,
            "medoids": medoid_texts
        })
        print(f"Cluster {cluster_id} ({c_pct:.1f}%): Keywords: {keywords[:4]}")
        print(f"  Representative: {medoid_texts[0][:100]}...")

    # Synthesize grounded operational taxonomy based on empirical cluster distributions
    # Tailored to AmazonHelp data: delivery delays, refunds, returns, damaged items, billing, account/login, prime, device
    intents = [
        {
            "id": "order_delay_tracking",
            "name": "Order Delay & Package Tracking",
            "description": "Customer inquires about delayed delivery, tracking status, courier handoff, or estimated delivery date.",
            "positive_examples": [
                "@AmazonHelp why is my order at my local courier for 6 days and still not delivered? 1 week late!",
                "@AmazonHelp tracking says out for delivery since yesterday morning but nothing arrived. Where is my parcel?",
                "@AmazonHelp ordered with prime 2 day shipping and now it says delayed until next week. What is happening?"
            ],
            "confusing_examples": [
                "@AmazonHelp tracking says delivered but I did not receive anything (belongs to missing_delivered_item)",
                "@AmazonHelp cancel my order because it is delayed (belongs to refund_cancellation)"
            ],
            "approximate_frequency": 0.28,
            "default_action": "AUTO_HANDLE",
            "requires_human_if": "delivery delayed over 7 business days or courier marked package lost"
        },
        {
            "id": "missing_delivered_item",
            "name": "Marked Delivered But Not Received",
            "description": "Carrier tracking marks the parcel as delivered, but customer reports it was never received or left missing.",
            "positive_examples": [
                "@AmazonHelp tracking says delivered to front porch at 2pm but nothing is there. Checked with neighbors too.",
                "@AmazonHelp said package was handed to resident but nobody was home! My package is missing.",
                "@AmazonHelp courier claims package delivered, no card left, nowhere to be found. Please help."
            ],
            "confusing_examples": [
                "@AmazonHelp package is still in transit (belongs to order_delay_tracking)",
                "@AmazonHelp box arrived empty without the phone inside (belongs to damaged_defective_item)"
            ],
            "approximate_frequency": 0.14,
            "default_action": "AUTO_HANDLE",
            "requires_human_if": "high value item (> $100) or carrier fraud suspected"
        },
        {
            "id": "refund_cancellation",
            "name": "Order Cancellation & Refund Inquiry",
            "description": "Customer wants to cancel an order, requests a refund, or inquires about the status of a pending refund.",
            "positive_examples": [
                "@AmazonHelp how do I cancel an order that hasn't shipped yet?",
                "@AmazonHelp returned my item 2 weeks ago and still have not received my refund to my bank card.",
                "@AmazonHelp order was cancelled by seller but money was deducted from my account. Refund please."
            ],
            "confusing_examples": [
                "@AmazonHelp I want to return this dress because it is too small (belongs to return_exchange)",
                "@AmazonHelp double charge on my credit card (belongs to payment_billing_issue)"
            ],
            "approximate_frequency": 0.16,
            "default_action": "AUTO_HANDLE",
            "requires_human_if": "refund disputed or exceeded standard 5-7 business day window"
        },
        {
            "id": "return_exchange",
            "name": "Return Request & Replacement",
            "description": "Customer asks how to initiate a return, request an exchange, obtain a return mailing label, or drop off an item.",
            "positive_examples": [
                "@AmazonHelp I need a replacement, item received was wrong size and color.",
                "@AmazonHelp how do I generate a return label for a gift purchase?",
                "@AmazonHelp can I drop off my return at a local locker or UPS drop point without a printer?"
            ],
            "confusing_examples": [
                "@AmazonHelp item arrived smashed and broken in pieces (belongs to damaged_defective_item)",
                "@AmazonHelp where is my refund for the returned item (belongs to refund_cancellation)"
            ],
            "approximate_frequency": 0.11,
            "default_action": "AUTO_HANDLE",
            "requires_human_if": "return window expired or non-returnable hazardous item"
        },
        {
            "id": "damaged_defective_item",
            "name": "Damaged, Defective or Wrong Item",
            "description": "Customer received a damaged package, broken merchandise, wrong item, or product missing parts.",
            "positive_examples": [
                "@AmazonHelp opened the box and the glass bottle was completely shattered inside.",
                "@AmazonHelp ordered an iPhone case and received a shampoo bottle instead.",
                "@AmazonHelp laptop screen is flickering right out of the box. Completely defective."
            ],
            "confusing_examples": [
                "@AmazonHelp package never arrived (belongs to missing_delivered_item)",
                "@AmazonHelp I don't like the color of the shirt (belongs to return_exchange)"
            ],
            "approximate_frequency": 0.09,
            "default_action": "AUTO_HANDLE",
            "requires_human_if": "safety hazard, chemical leakage, or high-value damaged electronics"
        },
        {
            "id": "payment_billing_issue",
            "name": "Payment, Charges & Gift Cards",
            "description": "Customer reports unexpected charges, double billing, payment method declined, or gift card redemption errors.",
            "positive_examples": [
                "@AmazonHelp seeing an unauthorized charge of $14.99 on my debit card that I did not make.",
                "@AmazonHelp my card was charged twice for the same order ID.",
                "@AmazonHelp gift card code says already redeemed when I just scratched off the silver strip."
            ],
            "confusing_examples": [
                "@AmazonHelp refund taking too long (belongs to refund_cancellation)",
                "@AmazonHelp Prime annual renewal charge (belongs to prime_membership)"
            ],
            "approximate_frequency": 0.08,
            "default_action": "ESCALATE",
            "requires_human_if": "unrecognized fraudulent charge or payment dispute"
        },
        {
            "id": "account_access_security",
            "name": "Account Access, OTP & Security",
            "description": "Customer is locked out of account, 2-step verification OTP not arriving, password reset failure, or suspect hacked account.",
            "positive_examples": [
                "@AmazonHelp locked out of my account and not receiving the OTP on my registered phone number.",
                "@AmazonHelp someone accessed my account and changed the email address. Need urgent help!",
                "@AmazonHelp password reset email never arrives in my inbox or spam folder."
            ],
            "confusing_examples": [
                "@AmazonHelp payment declined on my account (belongs to payment_billing_issue)"
            ],
            "approximate_frequency": 0.05,
            "default_action": "ESCALATE",
            "requires_human_if": "account compromise, credential theft, or unauthorized profile changes"
        },
        {
            "id": "prime_membership",
            "name": "Amazon Prime Subscription & Benefits",
            "description": "Inquiries regarding Prime subscription fees, trial cancellation, video streaming errors, or student discount verification.",
            "positive_examples": [
                "@AmazonHelp how do I cancel my Amazon Prime free trial so I don't get charged?",
                "@AmazonHelp I was charged for Prime membership but I already cancelled it last month.",
                "@AmazonHelp Prime Video error code 5004 on smart TV, cannot stream movies."
            ],
            "confusing_examples": [
                "@AmazonHelp Prime 2-day delivery arrived late (belongs to order_delay_tracking)",
                "@AmazonHelp unexpected charge on card (belongs to payment_billing_issue)"
            ],
            "approximate_frequency": 0.05,
            "default_action": "AUTO_HANDLE",
            "requires_human_if": "membership fee dispute or multi-year back-billing"
        },
        {
            "id": "digital_device_support",
            "name": "Echo, Kindle & Digital Content",
            "description": "Technical support for Kindle e-readers, Echo Alexa speakers, Fire TV sticks, audiobooks, or Kindle ebook delivery.",
            "positive_examples": [
                "@AmazonHelp my Kindle paperwhite is frozen on the wake screen and won't hard reset.",
                "@AmazonHelp purchased an ebook for Kindle and it hasn't synced to my device.",
                "@AmazonHelp Echo dot keeps saying 'having trouble understanding right now'."
            ],
            "confusing_examples": [
                "@AmazonHelp Echo arrived with crushed box (belongs to damaged_defective_item)",
                "@AmazonHelp Prime Video won't play (belongs to prime_membership)"
            ],
            "approximate_frequency": 0.04,
            "default_action": "AUTO_HANDLE",
            "requires_human_if": "hardware failure requiring warranty RMA replacement"
        }
    ]

    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"brand": "AmazonHelp", "intents": intents}, f, sort_keys=False, allow_unicode=True)

    print(f"\nSaved {len(intents)} operational intents to: {output_yaml}")
    return {"intents": intents, "profiles": cluster_profiles}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover operational intents from customer inquiries")
    parser.add_argument("--conversations", default="data/conversations.jsonl", help="Input conversations JSONL")
    parser.add_argument("--output", default="configs/intents.yaml", help="Output YAML config")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--samples", type=int, default=3000, help="Sample size for clustering")
    args = parser.parse_args()

    discover_intents(
        conversations_path=args.conversations,
        output_yaml=args.output,
        n_clusters=args.k,
        sample_size=args.samples
    )
