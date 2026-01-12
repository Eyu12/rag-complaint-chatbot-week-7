"""
OPTIMIZED RAG Pipeline - Minimal Runtime Version
Focuses on core functionality with minimal dependencies
"""

import time
import json
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class FastRAGPipeline:
    """
    Optimized RAG pipeline for quick testing
    Uses lightweight models and minimal processing
    """

    def __init__(self, vector_store_path: str = 'vector_store/chroma_db'):
        """Initialize with minimal setup"""
        print("Initializing FAST RAG Pipeline")

        # Lightweight embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")

        # Load persistent ChromaDB
        self.client = chromadb.PersistentClient(
            path=vector_store_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_collection("complaint_chunks")
        print(f"Vector store loaded: {self.collection.count()} documents")

        # Simple templates (no LLM for speed)
        self.templates = {
            'credit_card': "Based on {count} complaints about credit cards, common issues include: {issues}",
            'loan': "From {count} loan complaints, customers report: {issues}",
            'general': "Analysis of {count} complaints shows: {issues}"
        }

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Fast and safe retrieval"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )

        documents = results.get('documents')
        metadatas = results.get('metadatas')
        distances = results.get('distances')

        # Guard against None or empty results
        if not documents or not metadatas or not distances:
            return []

        retrieved: List[Dict[str, Any]] = []

        for i in range(len(documents[0])):
            retrieved.append({
                'text': documents[0][i],
                'metadata': metadatas[0][i] or {},
                'score': 1 - distances[0][i]
            })

        return retrieved

    def generate_simple_answer(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """Generate answer without LLM (maximum speed)"""
        if not docs:
            return "No relevant complaints found."

        # Extract product types
        products = {
            doc['metadata'].get('product_category', 'Unknown')
            for doc in docs
        }

        # Count issues
        issue_counts: Dict[str, int] = {}
        for doc in docs:
            issue = doc['metadata'].get('issue', 'Unknown')
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Top issues
        top_issues = sorted(
            issue_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        issue_list = ", ".join(
            f"{issue} ({count})" for issue, count in top_issues
        )

        # Select template
        query_lower = query.lower()
        if 'credit' in query_lower or 'card' in query_lower:
            template = self.templates['credit_card']
        elif 'loan' in query_lower:
            template = self.templates['loan']
        else:
            template = self.templates['general']

        return template.format(
            count=len(docs),
            issues=issue_list,
            products=", ".join(products)
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Complete fast query pipeline"""
        start_time = time.time()

        docs = self.retrieve(question, k=3)
        answer = self.generate_simple_answer(question, docs)

        # Prepare sources (top 2 only)
        sources = []
        for doc in docs[:2]:
            sources.append({
                'product': doc['metadata'].get('product_category', 'Unknown'),
                'issue': doc['metadata'].get('issue', 'Unknown'),
                'excerpt': doc['text'][:150] + '...',
                'score': round(doc['score'], 3)
            })

        elapsed_time = round(time.time() - start_time, 2)

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'retrieved': len(docs),
            'time': elapsed_time
        }


def run_fast_evaluation() -> List[Dict[str, Any]]:
    """Quick evaluation (completes in under 1 minute)"""
    print("\nFAST EVALUATION (5 QUESTIONS)")
    print("=" * 60)

    rag = FastRAGPipeline()

    questions = [
        "What are credit card complaints?",
        "Loan issues?",
        "Bank account problems?",
        "Customer service complaints?",
        "Fraud reports?"
    ]

    results: List[Dict[str, Any]] = []

    for idx, question in enumerate(questions, start=1):
        print(f"\nQ{idx}: {question}")
        result = rag.query(question)

        print(f"Answer (preview): {result['answer'][:80]}...")
        print(f"Sources: {result['retrieved']} | Time: {result['time']}s")

        results.append({
            'Question': question,
            'Answer Length': len(result['answer']),
            'Sources Retrieved': result['retrieved'],
            'Time (s)': result['time']
        })

    avg_time = np.mean([r['Time (s)'] for r in results])
    avg_sources = np.mean([r['Sources Retrieved'] for r in results])

    print("\nSUMMARY")
    print(f"Average response time: {avg_time:.2f}s")
    print(f"Average sources retrieved: {avg_sources:.1f}")
    print(f"Total evaluation time: {sum(r['Time (s)'] for r in results):.2f}s")

    return results


if __name__ == "__main__":
    results = run_fast_evaluation()

    # Save report
    with open('data/fast_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\nTask 3 FAST evaluation completed")
    print("Report saved to: data/fast_evaluation.json")
