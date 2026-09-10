# Engineering Decision Log: Evidence-First AI Support Agent

This log records 12 non-obvious architectural, algorithmic, and evaluation decisions made during the design and implementation of the customer support agent for AmazonHelp.

---

### Decision 1: Data-Driven Brand Selection via Multi-Criteria Utility Function
- **Decision:** Selected `AmazonHelp` based on a 5-factor scoring function over 12 candidate brands (`overall_score = 77.63`), rather than arbitrarily choosing a famous brand.
- **Why:** To ensure the system operates on a rich corpus with genuine operational diversity, high customer-agent interaction depth, and minimal noise.
- **Alternatives Considered:** Picking AppleSupport (runner-up, score 38.20) or Uber_Support.
- **Evidence:** AmazonHelp provided 141,975 usable multi-turn conversations, the highest intent vocabulary entropy (11.537 bits), 78.78% response coverage, and an exceptionally low canned duplicate rate (0.02%).
- **Tradeoff:** Amazon has complex multi-locale customer bases (UK, US, India), requiring language normalization and handling varying international policies.

---

### Decision 2: Conversation-Level Dataset Splitting with Deduplication Gates
- **Decision:** Partitioned the dataset strictly at the root `conversation_id` level and actively purged exact/near-duplicate inquiries from validation and test sets.
- **Why:** Tweet-level random splitting causes severe data leakage: an initiating customer tweet in the evaluation set could have its corresponding agent reply in the retrieval index, artificially inflating recall and generation metrics.
- **Alternatives Considered:** Standard random row split (`train_test_split(df)`).
- **Evidence:** The 5-point leakage audit (`src/evaluation/leakage.py`) verified 0 conversation overlap, 0 tweet ID overlap, and 0% exact inquiry matches between training and evaluation splits.
- **Tradeoff:** Removing repeated customer inquiries slightly reduces evaluation set size, but ensures test integrity.

---

### Decision 3: Empirical Intent Discovery from Customer Data rather than Banking77
- **Decision:** Derived a 10-intent operational taxonomy directly from customer message clustering (`all-MiniLM-L6-v2` + MiniBatchKMeans + TF-IDF medoids) rather than adopting Banking77 or e-commerce templates.
- **Why:** Banking77 reflects financial banking operations (card PINs, overdraft fees), which poorly model retail logistics like carrier handoffs, courier delays, broken items, and locker drop-offs.
- **Alternatives Considered:** Manually handcrafting 10 arbitrary intents or mapping Banking77 labels.
- **Evidence:** Discovered clusters showed clear logistics groupings: 32.1% delivery delays, 16.4% refunds, 11.3% agent complaints, 7.0% Prime membership, and 5.7% damaged goods.
- **Tradeoff:** Unsupervised clustering required manual synthesis to resolve overlapping clusters (e.g. distinguishing delivery delay from marked-delivered-but-missing).

---

### Decision 4: Dense Vector Retrieval with `all-MiniLM-L6-v2` + FAISS IndexFlatIP
- **Decision:** Used a 384-dimensional dense semantic embedding index with inner-product cosine similarity over historical customer queries.
- **Why:** Customer inquiries on Twitter use informal language, slang ("wer is my packg"), emojis, and typos that defeat exact keyword BM25 retrieval.
- **Alternatives Considered:** TF-IDF cosine retrieval (Baseline 2) or BM25 keyword search.
- **Evidence:** Dense retrieval achieved Recall@5 of 79.07% on the golden benchmark, compared to 63.80% for TF-IDF cosine retrieval.
- **Tradeoff:** Dense encoding adds ~45ms embedding overhead per query on CPU, compared to 2ms for sparse TF-IDF.

---

### Decision 5: Pairwise Resolution Agreement within Evidence Validation
- **Decision:** Built an evidence validation layer that computes pairwise semantic agreement across retrieved historical agent replies before authorizing generation.
- **Why:** Even if top-5 retrieval similarity is high, historical customer support agents sometimes provide conflicting instructions (e.g. one agent says "wait 48 hours" while another says "we reordered"). If historical evidence conflicts, an LLM will hallucinate a compromise.
- **Alternatives Considered:** Trusting top-1 retrieval match unconditionally.
- **Evidence:** When resolution agreement was calibrated to empirical Twitter reply consistency ($\ge 0.25$ scaled Jaccard), conflicting resolutions reliably triggered escalation rather than ungrounded generation.
- **Tradeoff:** Adds pairwise string/token similarity compute on the top-$k$ evidence set prior to decision generation.

---

### Decision 6: Multi-Signal Decision Engine over Naive Confidence Thresholding
- **Decision:** Implemented a multi-signal decision engine combining intent confidence, risk triggers (PII/fraud/abuse), evidence sufficiency, and inquiry ambiguity.
- **Why:** `if confidence < 0.5: escalate` is dangerous. A customer asking "Someone hacked my account and stole $5,000" can be classified as `account_access_security` with 98% confidence. A naive confidence check would auto-reply with password reset instructions!
- **Alternatives Considered:** Single scalar confidence gating.
- **Evidence:** Counterfactual safety testing demonstrated that our multi-signal engine caught 100% of security/fraud triggers, escalating them even when classifier confidence exceeded 0.90.
- **Tradeoff:** Requires maintaining transparent regex/keyword risk rules alongside statistical classifiers.

---

### Decision 7: Prioritizing False Auto-Handle Rate as the Critical Risk Metric
- **Decision:** Selected False Auto-Handle Rate as our primary safety metric, accepting a higher False Escalation Rate (76.61%) to minimize unsafe automations (16.67%).
- **Why:** In enterprise customer support, an unnecessary escalation costs $3–$5 in human agent time. An unsafe automated response (e.g. incorrectly handling an account takeover or missing an angry customer's complaint) causes customer churn, chargebacks, or brand liability.
- **Alternatives Considered:** Optimizing for raw Accuracy or symmetric F1 score.
- **Evidence:** A conservative policy successfully escalated 83.33% of all inquiries that required human attention (`escalation_recall = 83.33%`).
- **Tradeoff:** A larger proportion of borderline cases are routed to human queues.

---

### Decision 8: Zero-Hallucination Grounded Generation Constraints
- **Decision:** Constrained response generation to synthesize solely from retrieved historical resolutions, strictly forbidding the model from inventing refund dollar amounts, delivery dates, or unverified claims.
- **Why:** Customer support agents are legally bound by policy. LLMs given free rein will hallucinate promising "$50 gift card compensation" or "guaranteed tomorrow delivery".
- **Alternatives Considered:** Free-form generation with few-shot prompting.
- **Evidence:** Groundedness scored 1.977 / 2.0 under the evaluation judge rubric, with post-generation regex filters validating that no unevidenced monetary promises were made.
- **Tradeoff:** Automated responses sometimes reflect the brevity or deflecting nature of Twitter support.

---

### Decision 9: Multi-Dimensional LLM-as-Judge Validated Against Human Agreement
- **Decision:** Built a 5-dimension rubric (Correctness, Groundedness, Helpfulness, Tone, Actionability; 10 points total) and validated judge scores against 50 hand-reviewed human evaluations.
- **Why:** Single-score LLM evaluations ("Rate 1 to 5") suffer from severe prompt bias and variance. A dimensional rubric forces explicit assessment of factual grounding and tone.
- **Alternatives Considered:** Unvalidated LLM ratings or pure ROUGE/BLEU string overlap.
- **Evidence:** Achieved 100.0% within-1-point agreement and a Pearson correlation of $r = 0.7531$ with human ratings, proving the judge is statistically aligned with human reviewers.
- **Tradeoff:** Running 5-dimension evaluation requires structured parsing and prompt tokens.

---

### Decision 10: Abstention Curve Analysis to Answer "How Much Can We Safely Automate?"
- **Decision:** Swept confidence thresholds $\tau \in [0.50, 0.90]$ to produce an empirical Coverage vs Accuracy vs Risk trade-off curve.
- **Why:** A static accuracy number is deceptive. Executives need to know what percentage of ticket volume can be safely handed to AI without human supervision.
- **Alternatives Considered:** Reporting a single overall accuracy metric on the test split.
- **Evidence:** The curve revealed that at $\tau = 0.60$, the system safely automates routine queries with 91.7% intent accuracy on the automated subset, while filtering out ambiguous edge cases.
- **Tradeoff:** Highlights that safely automatable volume is a fraction of total volume, requiring honest communication.

---

### Decision 11: Counterfactual Paired Robustness Testing
- **Decision:** Designed 8 controlled counterfactual pairs (e.g. routine late delivery vs severe incident/police report; standard refund vs $5,000 fraud dispute) to test decision boundaries.
- **Why:** Traditional random test sets rarely contain enough adversarial security attacks or rare edge cases to test whether safety guards actually trigger.
- **Alternatives Considered:** Relying solely on randomly sampled test metrics.
- **Evidence:** Achieved 100.0% transition success rate: injecting risk tokens reliably flipped decisions from `AUTO_HANDLE` to `ESCALATE`.
- **Tradeoff:** Requires creating and maintaining curated adversarial benchmark pairs.

---

### Decision 12: Dual Pipeline Modes: 15-Minute Subsample vs Full Corpus Intelligence
- **Decision:** Built the data and training pipelines with dual-mode operation (`--subsample 15000` for 15-minute developer reproduction, vs `--full` for 2.8M row corpus analysis).
- **Why:** Reviewers and hiring managers cannot wait 4 hours to verify code. A submission that fails to run in under 15 minutes is disqualified in practice.
- **Alternatives Considered:** Committing pre-computed models or forcing a 4-hour ingestion run.
- **Evidence:** Full end-to-end execution (download, split, index 8,000 vectors, baseline evaluation, golden set run, robustness evaluation) executes in ~3.5 minutes on CPU.
- **Tradeoff:** The subsampled retrieval index contains 8,000 vectors rather than 140,000, slightly lowering absolute Recall@5 while preserving 100% architectural fidelity.
