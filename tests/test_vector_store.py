"""
Unit tests for vector store building pipeline
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
sys.path.append('../src')

from vector_store import VectorStoreBuilder

class TestVectorStoreBuilder(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing"""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample complaint data
        self.sample_data = pd.DataFrame({
            'Product': ['Credit card', 'Personal loan', 'Mortgage', 
                       'Credit card', 'Personal loan', 'Other'],
            'cleaned_narrative': [
                'Unauthorized charge on my credit card for $500. I reported it immediately but the bank took 2 weeks to respond.',
                'My personal loan payment was processed late causing a $50 late fee. Customer service was unhelpful.',
                'Mortgage payment was not credited properly. The bank claims they never received it.',
                'Credit card interest rate increased without notification. This is unfair practice.',
                'Loan servicing issues with multiple payments not being recorded correctly.',
                'General banking complaint about account management.'
            ],
            'Issue': ['Unauthorized transaction', 'Late fee', 'Payment processing',
                     'Interest rate', 'Loan servicing', 'Account management'],
            'Company': ['Bank A', 'Bank B', 'Bank C', 'Bank A', 'Bank B', 'Bank C'],
            'State': ['CA', 'NY', 'TX', 'FL', 'CA', 'NY']
        })
        
        # Save sample data
        self.data_path = Path(self.temp_dir.name) / 'sample_complaints.csv'
        self.sample_data.to_csv(self.data_path, index=False)
        
        # Initialize builder with sample data
        self.builder = VectorStoreBuilder(
            data_path=str(self.data_path),
            embedding_model_name='all-MiniLM-L6-v2',
            chunk_size=200,  # Smaller for testing
            chunk_overlap=20
        )
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_load_data(self):
        """Test data loading"""
        df = self.builder.load_data()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 6)
        self.assertIn('cleaned_narrative', df.columns)
    
    def test_stratified_sampling(self):
        """Test stratified sampling"""
        self.builder.load_data()
        sampled_df = self.builder.stratified_sampling(sample_size=4, random_state=42)
        
        self.assertIsNotNone(sampled_df)
        self.assertEqual(len(sampled_df), 4)
        
        # Check that we have samples from different products
        unique_products = sampled_df['Product'].nunique()
        self.assertGreaterEqual(unique_products, 2)
    
    def test_chunk_complaints(self):
        """Test text chunking"""
        self.builder.load_data()
        self.builder.stratified_sampling(sample_size=3, random_state=42)
        
        chunks, metadata = self.builder.chunk_complaints()
        
        self.assertGreater(len(chunks), 0)
        self.assertEqual(len(chunks), len(metadata))
        
        # Check chunk sizes
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)  # Should respect chunk_size
        
        # Check metadata structure
        for meta in metadata:
            self.assertIn('complaint_id', meta)
            self.assertIn('product_category', meta)
            self.assertIn('chunk_index', meta)
    
    def test_generate_embeddings(self):
        """Test embedding generation"""
        self.builder.load_data()
        self.builder.stratified_sampling(sample_size=2, random_state=42)
        chunks, _ = self.builder.chunk_complaints()
        
        embeddings = self.builder.generate_embeddings(chunks, batch_size=2)
        
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[0], len(chunks))
        self.assertEqual(embeddings.shape[1], 384)  # MiniLM dimension
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=0.01))
    
    def test_chunking_parameters(self):
        """Test different chunking parameters"""
        # Test with larger chunk size
        builder_large = VectorStoreBuilder(
            data_path=str(self.data_path),
            chunk_size=1000,
            chunk_overlap=100
        )
        
        builder_large.load_data()
        builder_large.stratified_sampling(sample_size=2, random_state=42)
        chunks, _ = builder_large.chunk_complaints()
        
        # Should have fewer chunks with larger chunk size
        self.assertLessEqual(len(chunks), 4)
        
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1000)
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved through pipeline"""
        self.builder.load_data()
        self.builder.stratified_sampling(sample_size=3, random_state=42)
        chunks, metadata = self.builder.chunk_complaints()
        
        # Check that all metadata fields are present
        expected_fields = [
            'complaint_id', 'product_category', 'issue',
            'company', 'state', 'chunk_index', 'total_chunks'
        ]
        
        for meta in metadata:
            for field in expected_fields:
                self.assertIn(field, meta)
            
            # Check that indices are consistent
            self.assertLess(meta['chunk_index'], meta['total_chunks'])

class TestVectorStoreIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def test_complete_pipeline_chroma(self):
        """Test complete pipeline with ChromaDB"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal test data
            test_data = pd.DataFrame({
                'Product': ['Credit card', 'Personal loan'],
                'cleaned_narrative': [
                    'Test complaint about credit card charges.',
                    'Test complaint about loan servicing.'
                ],
                'Issue': ['Test issue 1', 'Test issue 2'],
                'Company': ['Test Bank', 'Test Bank'],
                'State': ['CA', 'NY']
            })
            
            data_path = Path(temp_dir) / 'test_data.csv'
            test_data.to_csv(data_path, index=False)
            
            # Run pipeline
            builder = VectorStoreBuilder(
                data_path=str(data_path),
                chunk_size=100,
                chunk_overlap=10
            )
            
            vector_store, chunks, metadata, embeddings = builder.run_complete_pipeline(
                sample_size=2,
                use_chroma=True,
                save_embeddings=False
            )
            
            # Verify results
            self.assertIsNotNone(vector_store)
            self.assertGreater(len(chunks), 0)
            self.assertEqual(len(chunks), len(metadata))
            self.assertEqual(embeddings.shape[0], len(chunks))
            
            # Test search functionality
            if hasattr(vector_store, 'query'):
                results = vector_store.query(
                    query_texts=['credit card'],
                    n_results=1
                )
                self.assertGreater(len(results['documents'][0]), 0)
    
    def test_error_handling(self):
        """Test error handling for missing data"""
        builder = VectorStoreBuilder(data_path='nonexistent.csv')
        
        with self.assertRaises(FileNotFoundError):
            builder.load_data()
        
        # Test with empty data
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / 'empty.csv'
            pd.DataFrame().to_csv(empty_path)
            
            builder = VectorStoreBuilder(data_path=str(empty_path))
            
            with self.assertRaises(ValueError):
                builder.load_data()

if __name__ == '__main__':
    unittest.main(verbosity=2)