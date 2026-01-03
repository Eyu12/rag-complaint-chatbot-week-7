import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ComplaintDataPreprocessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.stats = {}

    def load_data(self, columns=None):
        if not self.data_path:
            raise ValueError("No data path provided")

        print("Loading data...")
        try:
            # Automatically detect format by extension
            if self.data_path.endswith(".parquet"):
                self.df = pd.read_parquet(self.data_path, columns=columns)
            elif self.data_path.endswith(".csv"):
                self.df = pd.read_csv(self.data_path, usecols=columns)
            else:
                raise ValueError("Unsupported file format. Use .csv or .parquet")

            print(f"Data loaded successfully: {self.df.shape}")
            return self.df

        except Exception as e:
            print(f"Failed to load data: {e}")
            self.df = None
            return None

    def initial_eda(self):
        if self.df is None:
            print("No data loaded for EDA.")
            return None

        print("\n--- Dataset Info ---")
        print(f"Total complaints: {len(self.df)}")
        print(f"Columns: {self.df.columns.tolist()}")
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        missing_percent = 100 * missing / len(self.df)
        missing_df = pd.DataFrame({"missing_count": missing, "missing_percent": missing_percent})
        print(missing_df.sort_values("missing_count", ascending=False).head(10))
        self.stats['missing_summary'] = missing_df.to_dict()
        return True

    def analyze_product_distribution(self, product_col='product_category'):
        if self.df is None or product_col not in self.df.columns:
            print(f"Product column '{product_col}' not found.")
            return None
        dist = self.df[product_col].value_counts()
        self.stats['product_distribution'] = dist.to_dict()
        print(f"\nTop products:\n{dist.head(10)}")
        return dist

    def analyze_narrative_length(self, narrative_col='Consumer complaint narrative'):
        if self.df is None or narrative_col not in self.df.columns:
            print(f"Narrative column '{narrative_col}' not found.")
            return None

        self.df['narrative_length'] = self.df[narrative_col].fillna('').str.len()
        self.df['word_count'] = self.df[narrative_col].fillna('').str.split().apply(len)
        total = len(self.df)
        empty = (self.df[narrative_col].fillna('') == '').sum()
        non_empty = total - empty
        print(f"Total complaints: {total}, With narratives: {non_empty}, Without: {empty}")
        self.stats['narrative_stats'] = {
            'total': total,
            'with_narratives': non_empty,
            'without_narratives': empty
        }
        return True

    def clean_narrative_text(self, text):
        if pd.isna(text) or text == '':
            return ""
        text = str(text).lower()
        patterns = [
            r'xxxx', r'\d{10,}',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
        ]
        for p in patterns:
            text = re.sub(p, '[REDACTED]', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?\-]', '', text)
        return text.strip()

    def apply_text_cleaning(self, narrative_col='Consumer complaint narrative'):
        if self.df is None or narrative_col not in self.df.columns:
            print("Cannot clean text: dataframe or column missing")
            return None
        print("Cleaning narrative text...")
        self.df['cleaned_narrative'] = self.df[narrative_col].apply(self.clean_narrative_text)
        print("Cleaning completed.")
        return self.df

    def remove_empty_narratives(self):
        if self.df is None or 'cleaned_narrative' not in self.df.columns:
            print("Cannot filter empty narratives")
            return None
        before = len(self.df)
        self.df = self.df[self.df['cleaned_narrative'].str.strip() != ''].copy()
        after = len(self.df)
        print(f"Removed {before - after} empty narratives. Remaining: {after}")
        return self.df

    def save(self, output_path='data/processed/filtered_complaints.csv'):
        if self.df is None:
            print("Nothing to save: dataframe is None")
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(output_path, index=False)
        print(f"Data saved to {output_path}")
        # Save stats
        stats_path = output_path.parent / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Stats saved to {stats_path}")


# ==========================
# Run Task 1
# ==========================
if __name__ == "__main__":
    preprocessor = ComplaintDataPreprocessor("data/raw/complaints.csv")

    preprocessor.load_data()
    preprocessor.initial_eda()
    preprocessor.analyze_product_distribution()
    preprocessor.analyze_narrative_length()
    preprocessor.apply_text_cleaning()
    preprocessor.remove_empty_narratives()
    preprocessor.save("data/processed/filtered_complaints.csv")
