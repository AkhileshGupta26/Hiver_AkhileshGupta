"""Dense vector retrieval engine using SentenceTransformers and FAISS.

Indexes historical customer-support cases from the training split and provides
sub-millisecond top-k cosine similarity search with full case metadata.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.evaluation.leakage import clean_text_for_comparison

class DenseRetriever:
    """FAISS-based dense semantic retriever over historical customer inquiries."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_dir: str = "models/faiss_index"
    ):
        self.model_name = model_name
        self.index_dir = index_dir
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: int = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def build_index(self, train_data: List[Dict[str, Any]], batch_size: int = 256):
        """Build FAISS inner-product index on normalized embeddings."""
        self._load_model()
        print(f"Building dense index on {len(train_data):,} historical cases...")
        
        raw_inquiries = [d["customer_inquiry"] for d in train_data]
        cleaned_inquiries = [clean_text_for_comparison(t) for t in raw_inquiries]
        
        # Store metadata
        self.metadata = [
            {
                "conversation_id": d["conversation_id"],
                "inquiry": d["customer_inquiry"],
                "agent_response": d.get("agent_resolution", ""),
                "intent": d.get("intent", "unknown")
            }
            for d in train_data
        ]
        
        # Compute normalized embeddings for cosine similarity
        embeddings = self.model.encode(
            cleaned_inquiries,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype("float32")
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"FAISS index built: {self.index.ntotal:,} vectors indexed.")

    def save(self, directory: Optional[str] = None):
        target_dir = directory or self.index_dir
        os.makedirs(target_dir, exist_ok=True)
        
        index_path = os.path.join(target_dir, "cases.index")
        meta_path = os.path.join(target_dir, "metadata.json")
        
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"Retriever artifacts saved to: {target_dir}")

    def load(self, directory: Optional[str] = None):
        target_dir = directory or self.index_dir
        index_path = os.path.join(target_dir, "cases.index")
        meta_path = os.path.join(target_dir, "metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing index files in {target_dir}")
            
        self._load_model()
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"Retriever loaded: {self.index.ntotal:,} vectors ready.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k similar historical cases."""
        self._load_model()
        cleaned_query = clean_text_for_comparison(query)
        q_emb = self.model.encode([cleaned_query], normalize_embeddings=True).astype("float32")
        
        scores, indices = self.index.search(q_emb, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            case_info = dict(self.metadata[idx])
            case_info["similarity"] = round(float(score), 4)
            case_info["rank"] = rank + 1
            results.append(case_info)
            
        return results

def evaluate_retrieval(
    retriever: DenseRetriever,
    eval_data: List[Dict[str, Any]],
    k_list: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """Evaluate retrieval quality (Recall@K) based on intent consistency with historical cases."""
    print(f"Evaluating retrieval quality over {len(eval_data):,} test cases...")
    hits_at_k = {k: 0 for k in k_list}
    max_k = max(k_list)
    
    for item in eval_data:
        query = item.get("customer_inquiry") or item.get("message", "")
        expected_intent = item.get("intent") or item.get("expected_intent", "")
        
        retrieved = retriever.retrieve(query, top_k=max_k)
        retrieved_intents = [r["intent"] for r in retrieved]
        
        for k in k_list:
            if expected_intent in retrieved_intents[:k]:
                hits_at_k[k] += 1
                
    total = len(eval_data)
    recall_scores = {
        f"Recall@{k}": round(hits_at_k[k] / total, 4)
        for k in k_list
    }
    
    print("\n--- Dense Retrieval Evaluation Results ---")
    for metric, score in recall_scores.items():
        print(f"  {metric}: {score * 100:.2f}%")
        
    return recall_scores

if __name__ == "__main__":
    with open("data/train.jsonl", "r", encoding="utf-8") as f:
        train_cases = [json.loads(line) for line in f if line.strip()]
        
    retriever = DenseRetriever()
    retriever.build_index(train_cases)
    retriever.save()
    
    with open("data/test.jsonl", "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f if line.strip()]
        
    evaluate_retrieval(retriever, test_cases[:500])
