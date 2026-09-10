# Dataset Analysis Report: Customer Support on Twitter

**Source Dataset:** `thoughtvector/customer-support-on-twitter` (twcs.csv)  
**Analysis Date:** 2026-09-10 09:04:04 UTC  
**Evaluation Mode:** Full Corpus Inspection  

---

## 1. Executive Summary & Headline Dimensions

| Metric | Measured Value | Implications for Support Agent Architecture |
| :--- | :--- | :--- |
| **Total Rows (Tweets)** | **2,811,774** | Large scale corpus requiring efficient chunked ingestion & sampling. |
| **Inbound Tweets (Customer)** | **1,537,843 (54.69%)** | Customer inquiries, complaints, and follow-ups. |
| **Outbound Tweets (Brand/Agent)** | **1,273,931 (45.31%)** | Historical agent resolutions and triage instructions. |
| **Unique Authors** | **702,777** | Customer author IDs anonymized as integers; brands use verified handles. |
| **Unique Inbound Customers** | **702,669** | High diversity of customer accounts. |
| **Unique Brands / Companies** | **108** | Multi-industry dataset across tech, retail, telecom, and airlines. |
| **Exact Duplicate Message Rate** | **1.04%** | Many agent messages are canned/templated ("Please DM us your order ID"). |
| **Inbound Response Coverage** | **75.71%** | Proportion of customer messages with identifiable agent responses. |
| **Reconstructed Conversations** | **794,335** | Root-level conversation threads identified via graph linkage. |
| **Mean Conversation Length** | **3.52 turns** | Typically short exchanges (median: 2, 75th percentile: 4). |
| **Max Conversation Length** | **1390 turns** | Rare long escalation threads. |

---

## 2. Missing Value Analysis

| Column Name | Missing Values | Missing Percentage | Interpretation |
| :--- | :--- | :--- | :--- |
| `tweet_id` | 0 | 0.00% |
| `author_id` | 0 | 0.00% |
| `inbound` | 0 | 0.00% |
| `created_at` | 0 | 0.00% |
| `text` | 0 | 0.00% |
| `response_tweet_id` | 1,040,629 | 37.01% |
| `in_response_to_tweet_id` | 794,335 | 28.25% |

### Critical Observations on Missing Values:
1. `response_tweet_id` has high missingness (~37.0%) because leaf/terminal turns in a conversation have no further replies.
2. `in_response_to_tweet_id` is missing in ~28.3% of rows, exactly denoting **conversation root turns** (inbound tickets opened by customers or unsolicited broadcast tweets).
3. `text`, `tweet_id`, `author_id`, `inbound`, and `created_at` have **0 missing values**, confirming schema completeness.

---

## 3. Brand Activity & Distribution

Top 15 brand accounts by total volume in corpus:

| Brand Handle | Outbound Tweets |
| :--- | :--- |
| `AmazonHelp` | 169,840 |
| `AppleSupport` | 106,860 |
| `Uber_Support` | 56,270 |
| `SpotifyCares` | 43,265 |
| `Delta` | 42,253 |
| `Tesco` | 38,573 |
| `AmericanAir` | 36,764 |
| `TMobileHelp` | 34,317 |
| `comcastcares` | 33,031 |
| `British_Airways` | 29,361 |
| `SouthwestAir` | 28,977 |
| `VirginTrains` | 27,817 |
| `Ask_Spectrum` | 25,860 |
| `XboxSupport` | 24,557 |
| `sprintcare` | 22,381 |

---

## 4. Conversation Structure & Graph Reconstruction

- **Graph Topology:** Tweets link hierarchically via `in_response_to_tweet_id` (parent pointer) and `response_tweet_id` (child pointer).
- **Multi-response Branching:** Found **222,426** instances where a single tweet generated multiple replies (comma-separated IDs), representing branching conversations or multiple agent interactions.
- **Dangling References:** Found **3,862** instances where a tweet references an `in_response_to_tweet_id` outside the scraped snapshot. The conversation parser must gracefully treat these as detached roots or thread fragments.

---

## 5. Quality Problems, Noise & Leakage Hazards

1. **Canned Responses & Repetitive Templates:**
   - Over **1.04%** of messages are identical. Brands frequently reply with canned templates:
     `"We would like to look into this for you. Please send us a DM with your account number."`
   - **Mitigation:** Retrieval indexing must deduplicate responses and extract true resolution actions rather than repetitive DM invitations.
2. **Private Message (DM) Deflection:**
   - A substantial proportion of Twitter support exchanges end with the brand deflecting the user to Direct Messages for privacy/security reasons (e.g. sharing PII or passwords).
   - **Architectural Decision:** The agent must learn to categorize inquiries that genuinely require private info or escalation to human agents, matching brand historical policy.
3. **Data Leakage Vulnerability:**
   - If splitting is performed randomly at the tweet level, an agent response could be in the training/retrieval set while its initiating customer tweet is in the evaluation set.
   - **Mandatory Enforcement:** All dataset splitting must be **strictly grouped by `conversation_id`**. No conversation thread may cross train, validation, or test boundaries.
