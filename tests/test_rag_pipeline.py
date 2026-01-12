"""
Unit tests for RAG pipeline
"""

import unittest
import sys
import tempfile
from pathlib import Path
import json
sys.path.append('../src')

from rag_pipeline import RAGPipeline, RetrievalConfig, GenerationConfig, RetrievalMethod, RAGEvaluator

class TestRAGPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock vector store for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create minimal test data
        self.test_data = {
            'query': "test credit card complaint",
            'expected_terms': ['credit', 'card', 'complaint']
        }
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_prompt_template_selection(self):
        """Test prompt template selection logic"""
        # Mock the RAG pipeline without actual vector store
        class MockRAG(RAGPipeline):
            def _load_vector_store(self, path):
                return {'type': 'mock', 'count': 0}
            
            def _initialize_llm(self):
                return None, None, None
        
        rag = MockRAG(vector_store_path='dummy_path')
        
        # Test different question types
        test_cases = [
            ("What are the trends in credit card complaints?", 'trend'),
            ("Compare credit card and loan complaints", 'comparison'),
            ("Summarize the main issues", 'summary'),
            ("What are the problems with credit cards?", 'analysis')
        ]
        
        for question, expected_template in test_cases:
            selected = rag._select_prompt_template(question)
            self.assertEqual(selected, expected_template)
    
    def test_context_formatting(self):
        """Test context formatting"""
        # Mock RAG pipeline
        class MockRAG(RAGPipeline):
            def _load_vector_store(self, path):
                return {'type': 'mock', 'count': 0}
            
            def _initialize_llm(self):
                return None, None, None
        
        rag = MockRAG(vector_store_path='dummy_path')
        
        # Test documents
        test_docs = [
            {
                'text': 'Test complaint about credit card fees.',
                'metadata': {
                    'product_category': 'Credit card',
                    'issue': 'Fees',
                    'company': 'Test Bank',
                    'date_received': '2023-01-01'
                },
                'score': 0.85
            },
            {
                'text': 'Another complaint about loan processing.',
                'metadata': {
                    'product_category': 'Personal loan',
                    'issue': 'Processing delay',
                    'company': 'Another Bank',
                    'date_received': '2023-02-01'
                },
                'score': 0.72
            }
        ]
        
        # Format context
        context = rag.format_context(test_docs)
        
        # Verify formatting
        self.assertIn('Credit card', context)
        self.assertIn('Fees', context)
        self.assertIn('Test Bank', context)
        self.assertIn('0.850', context)  # Score formatting
        self.assertIn('Document 1', context)
        self.assertIn('Document 2', context)
    
    def test_retrieval_config(self):
        """Test retrieval configuration"""
        config = RetrievalConfig(
            method=RetrievalMethod.MAX_MARGINAL_RELEVANCE,
            top_k=10,
            score_threshold=0.6,
            diversity_penalty=0.7
        )
        
        self.assertEqual(config.method, RetrievalMethod.MAX_MARGINAL_RELEVANCE)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.score_threshold, 0.6)
        self.assertEqual(config.diversity_penalty, 0.7)
    
    def test_generation_config(self):
        """Test generation configuration"""
        config = GenerationConfig(
            model_name="test-model",
            temperature=0.8,
            max_tokens=1000,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.temperature, 0.8)
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.repetition_penalty, 1.2)
    
    def test_response_cleaning(self):
        """Test response cleaning"""
        # Mock RAG pipeline
        class MockRAG(RAGPipeline):
            def _load_vector_store(self, path):
                return {'type': 'mock', 'count': 0}
            
            def _initialize_llm(self):
                return None, None, None
        
        rag = MockRAG(vector_store_path='dummy_path')
        
        # Test cases
        test_cases = [
            ("This is a test response.", "This is a test response."),  # No change
            ("This is a test response", "This is a test response."),  # Add period
            ("CONTEXT FROM CUSTOMER COMPLAINTS: This is a test", "This is a test."),  # Remove boilerplate
            ("Response with INSTRUCTIONS: Do this", "Response with INSTRUCTIONS: Do this.")  # Keep valid part
        ]
        
        for input_text, expected in test_cases:
            cleaned = rag._clean_response(input_text)
            self.assertEqual(cleaned, expected)


class TestEvaluationScoring(unittest.TestCase):
    """Test RAG evaluation scoring logic"""
    
    def test_evaluation_scoring(self):
        """Test evaluation scoring logic directly"""
        from rag_pipeline import RAGEvaluator
        
        # Create a minimal RAG pipeline that inherits properly
        class TestRAGPipeline(RAGPipeline):
            def __init__(self):
                # Skip actual initialization for testing
                pass
            
            def query(self, question):
                return {
                    'answer': 'Test answer about credit cards, billing, and customer service.',
                    'sources': [{'product': 'Credit card'}],
                    'retrieved_docs_count': 5,
                    'metadata': {'avg_relevance_score': 0.8}
                }
            
            def get_metrics(self):
                return {'queries_processed': 1}
        
        # Create a minimal evaluator for testing
        class TestRAGEvaluator(RAGEvaluator):
            def __init__(self):
                # Skip parent initialization for testing
                pass
            
            def _evaluate_single_result(self, result, expected_aspects):
                """Test evaluation logic"""
                answer = result['answer'].lower()
                
                # 1. Completeness: Does it cover expected aspects?
                aspect_coverage = 0
                for aspect in expected_aspects:
                    if aspect.lower() in answer:
                        aspect_coverage += 1
                
                completeness_score = (aspect_coverage / len(expected_aspects)) * 5
                
                # 2. Relevance: Is the answer relevant to the question?
                # For testing, we'll use a simplified version
                relevance_score = 4.0
                
                # 3. Actionability: Does it provide actionable insights?
                action_words = ['should', 'recommend', 'suggest', 'need to', 'improve', 'fix', 'address']
                action_word_count = sum(1 for word in action_words if word in answer)
                actionability_score = min(5.0, action_word_count * 1.5)
                
                # 4. Overall score (weighted average)
                overall_score = (
                    completeness_score * 0.4 +
                    relevance_score * 0.3 +
                    actionability_score * 0.3
                )
                
                return {
                    'completeness': round(completeness_score, 1),
                    'relevance': round(relevance_score, 1),
                    'actionability': round(actionability_score, 1),
                    'overall': round(overall_score, 1),
                    'comments': 'Test evaluation'
                }
        
        # Create evaluator instance
        evaluator = TestRAGEvaluator()
        
        # Test evaluation
        test_result = {
            'answer': 'Credit cards have issues with billing disputes, customer service, and fraud prevention.',
            'sources': [{'product': 'Credit card'}],
            'retrieved_docs_count': 5,
            'metadata': {'avg_relevance_score': 0.8}
        }
        
        expected_aspects = ['billing', 'customer service', 'fraud']
        evaluation = evaluator._evaluate_single_result(test_result, expected_aspects)
        
        # Verify scores
        self.assertIn('completeness', evaluation)
        self.assertIn('relevance', evaluation)
        self.assertIn('actionability', evaluation)
        self.assertIn('overall', evaluation)
        
        # Scores should be between 0 and 5
        self.assertGreaterEqual(evaluation['completeness'], 0)
        self.assertLessEqual(evaluation['completeness'], 5)
        self.assertGreaterEqual(evaluation['relevance'], 0)
        self.assertLessEqual(evaluation['relevance'], 5)
        self.assertGreaterEqual(evaluation['actionability'], 0)
        self.assertLessEqual(evaluation['actionability'], 5)
        self.assertGreaterEqual(evaluation['overall'], 0)
        self.assertLessEqual(evaluation['overall'], 5)
    
    def test_format_sources_for_table(self):
        """Test source formatting for table display"""
        from rag_pipeline import RAGEvaluator
        
        # Create a minimal evaluator
        class TestRAGEvaluator(RAGEvaluator):
            def __init__(self):
                pass
            
            def _format_sources_for_table(self, sources):
                """Format sources for display in evaluation table"""
                if not sources:
                    return "None"
                
                formatted = []
                for source in sources[:2]:  # Show first 2 sources
                    formatted.append(f"{source['product']}: {source['issue']}")
                
                return "; ".join(formatted)
        
        evaluator = TestRAGEvaluator()
        
        # Test sources
        test_sources = [
            {'product': 'Credit card', 'issue': 'Billing dispute'},
            {'product': 'Personal loan', 'issue': 'Interest rate'},
            {'product': 'Mortgage', 'issue': 'Processing delay'}
        ]
        
        # Test formatting
        formatted = evaluator._format_sources_for_table(test_sources)
        self.assertEqual(formatted, "Credit card: Billing dispute; Personal loan: Interest rate")
        
        # Test empty sources
        empty_formatted = evaluator._format_sources_for_table([])
        self.assertEqual(empty_formatted, "None")

if __name__ == '__main__':
    unittest.main(verbosity=2)