import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('../src')

from data_preprocessing import ComplaintDataPreprocessor

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing"""
        self.sample_data = pd.DataFrame({
            'Date received': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'Product': ['Credit card', 'Personal loan', 'Mortgage', 'Other product'],
            'Consumer complaint narrative': [
                'I am writing to file a complaint about my credit card. Charge was unauthorized.',
                'Loan servicing issues with my personal loan. Interest rate problems.',
                'Mortgage payment processing delay.',
                'This is another type of complaint.'
            ],
            'Company': ['Bank A', 'Bank B', 'Bank C', 'Bank D'],
            'State': ['CA', 'NY', 'TX', 'FL']
        })
        
        self.preprocessor = ComplaintDataPreprocessor()
        
    def test_clean_narrative_text(self):
        """Test text cleaning function"""
        test_text = "I am writing to file a complaint. Account Number: 1234567890. XXXX issues."
        cleaned = self.preprocessor.clean_narrative_text(test_text)
        
        # Should remove boilerplate
        self.assertNotIn('i am writing to file a complaint', cleaned)
        # Should redact account numbers
        self.assertIn('[redacted]', cleaned.lower())
        # Should remove XXXX
        self.assertNotIn('xxxx', cleaned.lower())
        
    def test_filter_products(self):
        """Test product filtering"""
        # Simulate the filtering process
        target_products = ['Credit card', 'Personal loan', 'Mortgage']
        filtered = self.sample_data[self.sample_data['Product'].isin(target_products)]
        
        self.assertEqual(len(filtered), 3)
        self.assertNotIn('Other product', filtered['Product'].values)
        
    def test_empty_narrative_removal(self):
        """Test removal of empty narratives"""
        data_with_empty = pd.DataFrame({
            'Product': ['A', 'B', 'C'],
            'Consumer complaint narrative': ['Some text', '', np.nan]
        })
        
        # Count non-empty
        non_empty = data_with_empty[data_with_empty['Consumer complaint narrative'].notna() & 
                                   (data_with_empty['Consumer complaint narrative'] != '')]
        
        self.assertEqual(len(non_empty), 1)

if __name__ == '__main__':
    unittest.main()