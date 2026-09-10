"""Interactive Streamlit demo for the Hiver AI Customer Support Agent.

Enables reviewers to enter custom customer inquiries or test preset benchmark cases,
inspecting intent classification, risk assessment, dense retrieval evidence,
multi-signal decision logic, and grounded response generation.
"""

import sys
import os

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.pipeline import SupportAgentPipeline

st.set_page_config(
    page_title="Hiver Support Agent — Evidence-First AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Hiver AI Customer Support Agent")
st.markdown(
    "**Evidence-First, Trustworthy Customer Support Automation** built on real-world "
    "`AmazonHelp` interactions. Grounds replies in historical resolutions and conservatively "
    "escalates ambiguous, sensitive, or conflicting inquiries."
)

@st.cache_resource(show_spinner="Loading models and FAISS retrieval index...")
def get_pipeline():
    return SupportAgentPipeline()

pipeline = get_pipeline()

# Sidebar Presets
st.sidebar.header("🧪 Benchmark Preset Test Cases")
presets = {
    "Select a preset...": "",
    "Routine Delivery Delay (Auto-Handle)": "@AmazonHelp why is my order at the local courier for 4 days and still not delivered? Tracking says delayed.",
    "Broken Merchandise Replacement (Auto-Handle)": "@AmazonHelp opened my package and the coffee mug was completely shattered in pieces. How do I get a replacement?",
    "Hacked Account / Credential Theft (Escalate)": "@AmazonHelp Someone hacked into my Amazon account, changed my login email and phone number! Need help immediately!",
    "Unauthorized $5,000 Card Charge (Escalate)": "@AmazonHelp I see an unauthorized charge of $5,000 on my credit card from AMZN MKTP that I did not authorize!",
    "Customer Demanding Human Agent (Escalate)": "@AmazonHelp Stop sending me automated bot responses. Connect me to a real human supervisor right now.",
    "Underspecified Short Query (Clarify)": "@AmazonHelp late"
}

selected_preset = st.sidebar.selectbox("Choose an evaluation case:", list(presets.keys()))

default_text = presets[selected_preset] if selected_preset != "Select a preset..." else ""

# Input Form
with st.form(key="query_form"):
    user_input = st.text_area(
        "Enter Customer Message:",
        value=default_text or "@AmazonHelp my package was supposed to arrive yesterday but tracking hasn't updated in 48 hours.",
        height=100
    )
    submit_btn = st.form_submit_button("Run Support Agent Pipeline", type="primary")

if submit_btn and user_input.strip():
    with st.spinner("Processing through multi-signal decision engine..."):
        result = pipeline.process_message(user_input.strip())

    st.markdown("---")
    st.subheader("🎯 Agent Triage & Response")

    # Decision Banner
    decision = result["decision"]
    if decision == "AUTO_HANDLE":
        st.success(f"### Decision: ✅ AUTO-HANDLE ({result['action_code']})")
    elif decision == "ASK_CLARIFYING_QUESTION":
        st.warning(f"### Decision: ❓ ASK CLARIFYING QUESTION ({result['action_code']})")
    else:
        st.error(f"### Decision: 🚨 ESCALATE TO HUMAN AGENT ({result['action_code']})")

    # Generated Reply Box
    st.markdown("#### 💬 Generated Customer Reply")
    st.info(result["generated_response"])

    # Decision Rationale
    st.markdown("#### 📋 Stated Decision Rationale")
    for r in result["decision_reasons"]:
        st.markdown(f"- {r}")

    # Metrics Columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Intent", result["predicted_intent"])
        st.caption(f"Confidence: {result['intent_confidence']*100:.1f}%")
    with col2:
        risk_color = "red" if result["risk_tier"] != "LOW" else "green"
        st.metric("Risk Assessment", result["risk_tier"])
        st.caption(f"Triggers: {result['risk_triggers'] or 'None'}")
    with col3:
        st.metric("Evidence Coverage", f"{result['retrieval_confidence']*100:.1f}%")
        st.caption(f"Strong Precedents: {result['strong_evidence_matches']}")
    with col4:
        st.metric("Resolution Consensus", f"{result['resolution_agreement']*100:.1f}%")
        st.caption(f"Latency: {result['latency_ms']} ms")

    # Expandable Retrieved Precedents
    with st.expander("🔍 View Retrieved Historical Cases (Dense FAISS Evidence)", expanded=False):
        for idx, ev in enumerate(result["retrieved_evidence"]):
            st.markdown(f"**Precedent #{idx+1}** (Similarity: `{ev['similarity']:.3f}` | Intent: `{ev['intent']}`)")
            st.markdown(f"> *Historical Agent Resolution:* {ev['historical_response']}")
            st.caption(f"Conversation ID: `{ev['conversation_id']}`")
            st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### 📊 Benchmark Summary\n"
    "- **Intent Accuracy:** 48.8% (on 172-case Hard Golden Set)\n"
    "- **Retrieval Recall@5:** 79.1%\n"
    "- **Escalation Recall:** 83.3%\n"
    "- **False Auto-Handle Rate:** 16.7%\n"
    "- **Counterfactual Robustness:** 100.0%\n"
    "- **Judge-Human Agreement:** 100% (within 1-pt)\n"
)
