"""
Task 4: Interactive Chat Interface for CrediTrust Complaint Analysis System
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
from dataclasses import dataclass
import warnings
import socket

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# --- RAG pipeline imports ---
try:
    from rag_pipeline import RAGPipeline, RetrievalConfig, GenerationConfig, RetrievalMethod
    RAG_AVAILABLE = True
except ImportError:
    print("❌ Could not import RAGPipeline. Ensure rag_pipeline.py exists in src folder.")
    RAGPipeline = RetrievalConfig = GenerationConfig = RetrievalMethod = None
    RAG_AVAILABLE = False


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }


class CrediTrustChatbot:
    """Chatbot integrating RAG pipeline with conversation management and charts"""

    def __init__(self, vector_store_path: str = 'vector_store/chroma_db'):
        print("🚀 Initializing CrediTrust Complaint Analysis Chatbot")
        self.rag = self._initialize_rag_pipeline(vector_store_path)
        self.conversation_history: List[ChatMessage] = []
        self.current_session_id = f"session_{int(time.time())}"
        self.product_categories = self._load_product_categories()
        print(f"✅ Chatbot initialized. Session ID: {self.current_session_id}")
        if not RAG_AVAILABLE:
            print("⚠️ Running in dummy mode (RAGPipeline unavailable).")

    def _initialize_rag_pipeline(self, vector_store_path: str):
        if not RAG_AVAILABLE:
            return None
        retrieval_config = RetrievalConfig(
            method=RetrievalMethod.SEMANTIC,
            top_k=5,
            score_threshold=0.4
        )
        generation_config = GenerationConfig(model_name="microsoft/phi-2", max_tokens=500)
        return RAGPipeline(
            vector_store_path=vector_store_path,
            retrieval_config=retrieval_config,
            generation_config=generation_config
        )

    def _load_product_categories(self) -> List[str]:
        return ["Credit card", "Personal loan", "Mortgage",
                "Savings account", "Money transfer",
                "Debt collection", "Credit reporting"]

    # --- Query processing ---
    def process_query(self,
                      question: str,
                      product_filter: Optional[str] = None,
                      date_range: Optional[tuple] = None,
                      company_filter: Optional[str] = None,
                      show_sources: bool = True) -> Dict[str, Any]:

        if not question.strip():
            return {'answer': "Please enter a question about customer complaints.",
                    'sources': [], 'metadata': {}}

        filters = {}
        if product_filter and product_filter != "All Products":
            filters['product_category'] = product_filter
        if company_filter and company_filter != "All Companies":
            filters['company'] = company_filter
        if date_range:
            start, end = date_range
            filters['date_received'] = {'$gte': start.strftime('%Y-%m-%d'), '$lte': end.strftime('%Y-%m-%d')}

        try:
            if self.rag:
                result = self.rag.query(question, filters)
                answer = result.get('answer', '')
                sources = result.get('sources', [])
                metadata = result.get('metadata', {})
            else:
                # Dummy result
                answer = f"Simulated response to: {question}"
                sources = [{"title": "Example Doc", "text": "This is a dummy source.", "category": "Credit card",
                            "date": datetime.now(), "source": "Demo"}]
                metadata = {}
        except Exception as e:
            answer = f"Error processing query: {str(e)}"
            sources = []
            metadata = {}

        # Append conversation
        self.conversation_history.append(ChatMessage("user", question, datetime.now().isoformat()))
        self.conversation_history.append(ChatMessage("assistant", answer, datetime.now().isoformat()))

        return {'answer': answer, 'sources': sources, 'metadata': metadata}

    # --- Visualization ---
    def _create_comparison_chart(self, sources: List[Dict[str, Any]]) -> go.Figure:
        categories = [s.get('category', 'Unknown') for s in sources]
        counts = pd.Series(categories).value_counts()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values))
        fig.update_layout(title="Category Comparison")
        return fig

    def format_sources_display(self, sources: List[Dict[str, Any]]) -> str:
        html = "<ul>"
        for s in sources:
            html += f"<li><b>{s.get('title', 'No Title')}</b>: {s.get('text','')[:100]}...</li>"
        html += "</ul>"
        return html

    def create_visualization(self, result: Dict[str, Any]) -> Optional[go.Figure]:
        sources = result.get('sources', [])
        if not sources:
            return None
        return self._create_comparison_chart(sources)

    def get_conversation_summary(self) -> Dict[str, Any]:
        return {
            'total_queries': len([msg for msg in self.conversation_history if msg.role == 'user'])
        }

    def clear_conversation(self):
        self.conversation_history.clear()
        return "Conversation cleared."


# --- Gradio Interface ---
def create_interface():
    chatbot = CrediTrustChatbot()

    product_filter = gr.Dropdown(label="Product Category",
                                 choices=["All Products"] + chatbot.product_categories,
                                 value="All Products")
    company_filter = gr.Dropdown(label="Company",
                                 choices=["All Companies", "Bank of America", "Wells Fargo", "Chase", "Citibank"],
                                 value="All Companies")
    
    # Use Textbox for dates (YYYY-MM-DD)
    start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)",
                            value=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value=datetime.now().strftime('%Y-%m-%d'))

    chatbot_display = gr.Chatbot(label="Conversation", height=400)
    question_input = gr.Textbox(placeholder="Ask a question...", lines=2)
    submit_btn = gr.Button("Ask")
    clear_btn = gr.Button("Clear Chat")
    sources_display = gr.HTML("<div>Sources will appear here after your first query.</div>")
    visualization_display = gr.Plot()

    def process_message(question, product, company, start, end, history):
        if not question.strip():
            return history, "", "<div>Please enter a question.</div>", None

        # Parse date strings safely
        try:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
        except Exception:
            start_dt = datetime.now() - timedelta(days=365)
        try:
            end_dt = datetime.strptime(end, '%Y-%m-%d')
        except Exception:
            end_dt = datetime.now()

        result = chatbot.process_query(question, product_filter=product,
                                       company_filter=company, date_range=(start_dt, end_dt))
        history.append((question, result['answer']))
        sources_html = chatbot.format_sources_display(result['sources'])
        viz = chatbot.create_visualization(result)
        return history, "", sources_html, viz

    def clear_chat():
        chatbot.clear_conversation()
        return [], "", "<div>Chat cleared.</div>", None

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                question_input.render()
                submit_btn.render()
                clear_btn.render()
            with gr.Column():
                chatbot_display.render()
                visualization_display.render()
                sources_display.render()

        submit_btn.click(process_message,
                         inputs=[question_input, product_filter, company_filter, start_date, end_date, chatbot_display],
                         outputs=[chatbot_display, question_input, sources_display, visualization_display])

        clear_btn.click(clear_chat,
                        outputs=[chatbot_display, question_input, sources_display, visualization_display])

    return interface


# --- Auto port detection ---
def find_free_port(start_port: int = 7860, end_port: int = 7900) -> int:
    """Find the first free TCP port in the given range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start_port}-{end_port}")


def main():
    vector_store_path = Path("vector_store/chroma_db")
    if not vector_store_path.exists():
        print("❌ Vector store not found! Running in dummy mode.")

    interface = create_interface()

    # Launch Gradio on first available port
    port = find_free_port(7860, 7900)
    print(f"🌐 Launching Gradio app on http://127.0.0.1:{port}")
    interface.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()
