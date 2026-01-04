"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
for CrediTrust Financial Complaint Analysis System

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Text processing imports
from sentence_transformers import SentenceTransformer
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Vector store imports
import faiss
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Progress tracking
from tqdm.auto import tqdm
import time

class VectorStoreBuilder:
    """
    Build vector stores from complaint data for semantic search
    """
    
    def __init__(self, 
                 data_path='data/processed/filtered_complaints.csv',
                 embedding_model_name='all-MiniLM-L6-v2',
                 chunk_size=500,
                 chunk_overlap=50):
        """
        Initialize the vector store builder
        
        Args:
            data_path: Path to processed complaint data
            embedding_model_name: Name of sentence transformer model
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        self.data_path = Path(data_path)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.df: pd.DataFrame = pd.DataFrame()              
        self.sampled_df: pd.DataFrame = pd.DataFrame()   
        self.chunks: list = []                          
        self.metadata: list = []                        
        self.embeddings: np.ndarray = np.empty((0, 384))      
        
        # Initialize embedding model
        print(f" Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f" Model loaded. Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=['\n\n', '\n', '.', '!', '?', ',', ' ', '']
        )
        
    def load_data(self):
        """
        Load the processed complaint data
        """
        print(f" Loading processed data from: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Ensure required columns exist
        required_columns = ['cleaned_narrative']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        print(f" Data loaded successfully!")
        print(f"   Shape: {self.df.shape}")
        print(f"   Columns: {self.df.columns.tolist()}")
        
        # Show data overview
        if 'Product' in self.df.columns:
            print(f"\n Product distribution in loaded data:")
            product_dist = self.df['Product'].value_counts()
            for product, count in product_dist.head().items():
                print(f"   {product}: {count:,} complaints")
        
        return self.df
    
    def stratified_sampling(self, sample_size=12000, random_state=42):
        """
        Create stratified sample across product categories
        
        Args:
            sample_size: Target sample size
            random_state: Random seed for reproducibility
            
        Returns:
            pandas.DataFrame: Stratified sample
        """
        print(f"\n" + "="*60)
        print(f"🎯 CREATING STRATIFIED SAMPLE (n={sample_size:,})")
        print("="*60)
        
        if self.df is None:
            self.load_data()
        
        # Identify product column
        product_cols = [col for col in self.df.columns if 'product' in col.lower()]
        if not product_cols:
            raise ValueError("No product column found in data")
        
        product_col = product_cols[0]
        print(f" Using product column: '{product_col}'")
        
        # Get product distribution
        product_counts = self.df[product_col].value_counts()
        total_complaints = len(self.df)
        
        print(f"\n Original product distribution:")
        print(f"   Total complaints: {total_complaints:,}")
        print(f"   Unique products: {len(product_counts)}")
        
        # Calculate sampling proportions
        proportions = product_counts / total_complaints
        samples = []
        
        print(f"\n Sampling from each product category:")
        for product, proportion in proportions.items():
            # Calculate samples for this product
            n_samples = int(sample_size * proportion)
            
            # Ensure at least 1 sample for each product
            n_samples = max(1, n_samples)
            
            # Get product subset
            product_df = self.df[self.df[product_col] == product]
            
            # Ensure we don't sample more than available
            n_samples = min(n_samples, len(product_df))
            
            if n_samples > 0:
                # Sample without replacement
                sample = product_df.sample(n=n_samples, random_state=random_state)
                samples.append(sample)
                
                print(f"   {product}: {n_samples:,} samples ({len(product_df):,} available)")
        
        # Combine samples
        self.sampled_df = pd.concat(samples, ignore_index=True)
        
        # If we have fewer samples than target, sample additional from largest categories
        if len(self.sampled_df) < sample_size:
            additional_needed = sample_size - len(self.sampled_df)
            print(f"\n Sample size ({len(self.sampled_df):,}) less than target ({sample_size:,})")
            print(f"   Adding {additional_needed:,} additional samples...")
            
            # Get additional samples from largest categories
            remaining_df = self.df[~self.df.index.isin(self.sampled_df.index)]
            additional_sample = remaining_df.sample(n=additional_needed, random_state=random_state)
            self.sampled_df = pd.concat([self.sampled_df, additional_sample], ignore_index=True)
        
        # Verify sample distribution
        sample_product_dist = self.sampled_df[product_col].value_counts()
        print(f"\n Stratified sample created successfully!")
        print(f"   Sample size: {len(self.sampled_df):,}")
        print(f"   Sample distribution:")
        
        for product, count in sample_product_dist.items():
            sample_prop = count / len(self.sampled_df)
            original_prop = product_counts[product] / total_complaints
            print(f"   {product}: {count:,} ({sample_prop*100:.1f}%, original: {original_prop*100:.1f}%)")
        
        # Visualize sampling distribution
        self._visualize_sampling_distribution(product_counts, sample_product_dist, product_col)
        
        # Store sampling statistics
        sampling_stats = {
            'target_sample_size': sample_size,
            'actual_sample_size': len(self.sampled_df),
            'original_total': total_complaints,
            'original_products': len(product_counts),
            'sampling_distribution': sample_product_dist.to_dict(),
            'random_state': random_state
        }
        
        # Save sampling statistics
        stats_path = Path('data/processed/sampling_stats.json')
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(sampling_stats, f, indent=2)
        
        print(f"\n Sampling statistics saved to: {stats_path}")
        
        return self.sampled_df
    
    def _visualize_sampling_distribution(self, original_dist, sample_dist, product_col):
        """
        Visualize original vs sampled distribution
        """
        # Prepare data for visualization
        products = list(original_dist.index)[:10]  # Top 10 products
        
        original_counts = [original_dist[p] for p in products]
        sample_counts = [sample_dist.get(p, 0) for p in products]
        
        # Calculate proportions
        original_props = [c/sum(original_counts) for c in original_counts]
        sample_props = [c/len(self.sampled_df) for c in sample_counts]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original distribution (bar chart)
        axes[0, 0].barh(products[::-1], original_counts[::-1])
        axes[0, 0].set_xlabel('Number of Complaints')
        axes[0, 0].set_title('Original Distribution (Top 10 Products)')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Sample distribution (bar chart)
        axes[0, 1].barh(products[::-1], sample_counts[::-1])
        axes[0, 1].set_xlabel('Number of Complaints')
        axes[0, 1].set_title('Sample Distribution (Top 10 Products)')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Proportion comparison (line chart)
        x = range(len(products))
        axes[1, 0].plot(x, original_props, 'bo-', label='Original', linewidth=2)
        axes[1, 0].plot(x, sample_props, 'ro--', label='Sample', linewidth=2)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(products, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].set_title('Proportion Comparison: Original vs Sample')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Sample size by product
        product_names = list(sample_dist.index)
        sample_sizes = list(sample_dist.values)
        
        axes[1, 1].pie(sample_sizes[:10], labels=product_names[:10], autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Sample Composition (Top 10 Products)')
        
        plt.tight_layout()
        plt.savefig('data/sampling_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n Visualization saved to: data/sampling_distribution.png")
    
    def chunk_complaints(self, df=None, text_column='cleaned_narrative'):
        """
        Split complaint narratives into chunks
        
        Args:
            df: DataFrame containing complaints (uses sampled_df if None)
            text_column: Column containing text to chunk
            
        Returns:
            tuple: (chunks, metadata)
        """
        print(f"\n" + "="*60)
        print(f"  CHUNKING COMPLAINT NARRATIVES")
        print("="*60)
        
        if df is None:
            if self.sampled_df is None:
                raise ValueError("No data available. Run stratified_sampling() first.")
            df = self.sampled_df
        
        print(f" Chunking parameters:")
        print(f"   Chunk size: {self.chunk_size} characters")
        print(f"   Chunk overlap: {self.chunk_overlap} characters")
        print(f"   Text column: '{text_column}'")
        print(f"   Complaints to process: {len(df):,}")
        
        # Reset storage
        self.chunks = []
        self.metadata = []
        
        # Track statistics
        total_chunks = 0
        chunks_per_complaint = []
        
        print(f"\n Processing complaints...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
            complaint_id = row.get('Complaint ID', idx)
            narrative = row[text_column]
            
            # Skip empty narratives
            if pd.isna(narrative) or str(narrative).strip() == '':
                continue
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(str(narrative))
            
            # Store chunks and metadata
            for chunk_idx, chunk in enumerate(text_chunks):
                # Create metadata
                metadata = {
                    'complaint_id': str(complaint_id),
                    'product_category': str(row.get('Product', 'Unknown')),
                    'product': str(row.get('Sub-product', '')),
                    'issue': str(row.get('Issue', '')),
                    'sub_issue': str(row.get('Sub-issue', '')),
                    'company': str(row.get('Company', '')),
                    'state': str(row.get('State', '')),
                    'date_received': str(row.get('Date received', '')),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks),
                    'text_length': len(chunk),
                    'original_complaint_index': idx
                }
                
                self.chunks.append(chunk)
                self.metadata.append(metadata)
            
            chunks_per_complaint.append(len(text_chunks))
            total_chunks += len(text_chunks)
        
        print(f"\n Chunking completed!")
        print(f"   Total chunks created: {total_chunks:,}")
        print(f"   Average chunks per complaint: {np.mean(chunks_per_complaint):.2f}")
        print(f"   Min chunks per complaint: {np.min(chunks_per_complaint)}")
        print(f"   Max chunks per complaint: {np.max(chunks_per_complaint)}")
        print(f"   Chunk size range: {min([len(c) for c in self.chunks])} - {max([len(c) for c in self.chunks])} characters")
        
        # Visualize chunking results
        self._visualize_chunking_results(chunks_per_complaint)
        
        # Save chunking statistics
        chunking_stats = {
            'total_chunks': total_chunks,
            'total_complaints': len(df),
            'avg_chunks_per_complaint': float(np.mean(chunks_per_complaint)),
            'min_chunks_per_complaint': int(np.min(chunks_per_complaint)),
            'max_chunks_per_complaint': int(np.max(chunks_per_complaint)),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'chunk_length_stats': {
                'min': int(min([len(c) for c in self.chunks])),
                'max': int(max([len(c) for c in self.chunks])),
                'mean': float(np.mean([len(c) for c in self.chunks])),
                'median': float(np.median([len(c) for c in self.chunks]))
            }
        }
        
        stats_path = Path('data/processed/chunking_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(chunking_stats, f, indent=2)
        
        print(f"\n Chunking statistics saved to: {stats_path}")
        
        return self.chunks, self.metadata
    
    def _visualize_chunking_results(self, chunks_per_complaint):
        """
        Visualize chunking statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution of chunks per complaint
        axes[0, 0].hist(chunks_per_complaint, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(chunks_per_complaint), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(chunks_per_complaint):.2f}')
        axes[0, 0].set_xlabel('Chunks per Complaint')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Chunks per Complaint')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Chunk length distribution
        chunk_lengths = [len(c) for c in self.chunks]
        axes[0, 1].hist(chunk_lengths, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(np.mean(chunk_lengths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(chunk_lengths):.0f} chars')
        axes[0, 1].axvline(self.chunk_size, color='blue', linestyle=':',
                          label=f'Target: {self.chunk_size} chars')
        axes[0, 1].set_xlabel('Chunk Length (characters)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Chunk Lengths')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Cumulative distribution of chunks
        sorted_chunks = np.sort(chunks_per_complaint)
        cumulative = np.cumsum(sorted_chunks) / np.sum(chunks_per_complaint)
        axes[1, 0].plot(sorted_chunks, cumulative, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Chunks per Complaint')
        axes[1, 0].set_ylabel('Cumulative Proportion of Total Chunks')
        axes[1, 0].set_title('Cumulative Distribution of Chunks')
        axes[1, 0].grid(alpha=0.3)
        
        # Box plot of chunk lengths
        axes[1, 1].boxplot(chunk_lengths, vert=False)
        axes[1, 1].set_xlabel('Chunk Length (characters)')
        axes[1, 1].set_title('Box Plot of Chunk Lengths')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/chunking_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Chunking visualization saved to: data/chunking_analysis.png")
    
    def generate_embeddings(self, chunks=None, batch_size=32):
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of text chunks (uses self.chunks if None)
            batch_size: Batch size for embedding generation
            
        Returns:
            numpy.ndarray: Embedding vectors
        """
        print(f"\n" + "="*60)
        print(f" GENERATING EMBEDDINGS")
        print("="*60)
        
        if chunks is None:
            if not self.chunks:
                raise ValueError("No chunks available. Run chunk_complaints() first.")
            chunks = self.chunks
        
        print(f" Embedding parameters:")
        print(f"   Model: {self.embedding_model_name}")
        print(f"   Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        print(f"   Total chunks: {len(chunks):,}")
        print(f"   Batch size: {batch_size}")
        
        # Generate embeddings
        print(f"\n⚡ Generating embeddings...")
        start_time = time.time()
        
        self.embeddings = self.embedding_model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n Embeddings generated successfully!")
        print(f"   Time taken: {elapsed_time:.2f} seconds")
        print(f"   Speed: {len(chunks)/elapsed_time:.0f} chunks/second")
        print(f"   Embedding shape: {self.embeddings.shape}")
        print(f"   Memory usage: {self.embeddings.nbytes / 1024**2:.2f} MB")
        
        # Analyze embedding quality
        self._analyze_embeddings()
        
        # Save embeddings
        self._save_embeddings()
        
        return self.embeddings
    
    def _analyze_embeddings(self):
        """
        Analyze embedding quality and characteristics
        """
        print(f"\n Analyzing embedding quality...")
        
        # Calculate norms
        norms = np.linalg.norm(self.embeddings, axis=1)
        
        # Calculate pairwise distances sample
        sample_size = min(1000, len(self.embeddings))
        sample_indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
        sample_embeddings = self.embeddings[sample_indices]
        
        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(sample_embeddings)
        np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarities
        
        # Statistics
        stats = {
            'norm_mean': float(np.mean(norms)),
            'norm_std': float(np.std(norms)),
            'norm_min': float(np.min(norms)),
            'norm_max': float(np.max(norms)),
            'similarity_mean': float(np.mean(similarity_matrix)),
            'similarity_std': float(np.std(similarity_matrix)),
            'similarity_min': float(np.min(similarity_matrix)),
            'similarity_max': float(np.max(similarity_matrix)),
            'embedding_dimension': self.embeddings.shape[1],
            'total_embeddings': self.embeddings.shape[0]
        }
        
        print(f" Embedding statistics:")
        print(f"   Norm - Mean: {stats['norm_mean']:.4f}, Std: {stats['norm_std']:.4f}")
        print(f"   Similarity - Mean: {stats['similarity_mean']:.4f}, Std: {stats['similarity_std']:.4f}")
        print(f"   All embeddings are normalized: {np.allclose(norms, 1.0, atol=1e-3)}")
        
        # Visualize embeddings
        self._visualize_embeddings(norms, similarity_matrix)
        
        # Save statistics
        stats_path = Path('data/processed/embedding_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n Embedding statistics saved to: {stats_path}")
    
    def _visualize_embeddings(self, norms, similarity_matrix):
        """
        Visualize embedding characteristics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution of embedding norms
        axes[0, 0].hist(norms, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(1.0, color='red', linestyle='--', label='Ideal (1.0)')
        axes[0, 0].axvline(np.mean(norms), color='green', linestyle='--', 
                          label=f'Mean: {np.mean(norms):.4f}')
        axes[0, 0].set_xlabel('Embedding Norm')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Embedding Norms')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Distribution of cosine similarities
        flat_similarities = similarity_matrix.flatten()
        axes[0, 1].hist(flat_similarities, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(np.mean(flat_similarities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(flat_similarities):.4f}')
        axes[0, 1].set_xlabel('Cosine Similarity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Pairwise Similarities')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Scatter plot of first two PCA components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        sample_size = min(500, len(self.embeddings))
        sample_indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
        sample_embeddings = self.embeddings[sample_indices]
        
        # Color by product category if available
        if self.metadata:
            sample_metadata = [self.metadata[i] for i in sample_indices]
            # Get unique product categories
            product_categories = list(set([m.get('product_category', 'Unknown') for m in sample_metadata]))
            category_to_color = {cat: i for i, cat in enumerate(product_categories)}
            colors = [category_to_color[m.get('product_category', 'Unknown')] for m in sample_metadata]
            
            pca_result = pca.fit_transform(sample_embeddings)
            scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], c=colors, 
                                        cmap='tab20', alpha=0.6, s=20)
            axes[1, 0].set_xlabel('PCA Component 1')
            axes[1, 0].set_ylabel('PCA Component 2')
            axes[1, 0].set_title('PCA of Embeddings (Colored by Product Category)')
            
            # Add legend for top categories
            from matplotlib.lines import Line2D
            top_categories = product_categories[:5]  # Show top 5
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=plt.get_cmap("tab20")(i / 5), 
                                     label=cat, markersize=8) 
                             for i, cat in enumerate(top_categories)]
            axes[1, 0].legend(handles=legend_elements, title='Product Categories', 
                             bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Variance explained by PCA components
        pca_full = PCA().fit(self.embeddings[:1000])  # Fit on sample
        explained_variance = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        axes[1, 1].plot(range(1, len(explained_variance) + 1), cumulative_variance, 'b-', linewidth=2)
        axes[1, 1].axhline(0.8, color='r', linestyle='--', label='80% variance')
        axes[1, 1].axhline(0.9, color='g', linestyle='--', label='90% variance')
        axes[1, 1].set_xlabel('Number of Principal Components')
        axes[1, 1].set_ylabel('Cumulative Explained Variance')
        axes[1, 1].set_title('PCA Variance Explained')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../data/embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Embedding visualization saved to: data/embedding_analysis.png")
    
    def _save_embeddings(self, output_path='data/processed/embeddings.npy'):
        """
        Save embeddings to file
        
        Args:
            output_path: Path to save embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, self.embeddings)
        print(f" Embeddings saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024**2:.2f} MB")
    
    def build_faiss_index(self, embeddings=None, index_type='FlatL2'):
        """
        Build FAISS vector index
        
        Args:
            embeddings: Embedding vectors (uses self.embeddings if None)
            index_type: Type of FAISS index ('FlatL2' or 'IVFFlat')
            
        Returns:
            faiss.Index: FAISS index
        """
        print(f"\n" + "="*60)
        print(f"  BUILDING FAISS INDEX")
        print("="*60)
        
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings available. Run generate_embeddings() first.")
            embeddings = self.embeddings
        
        print(f" FAISS index parameters:")
        print(f"   Index type: {index_type}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   Number of vectors: {embeddings.shape[0]:,}")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        if index_type == 'FlatL2':
            # Simple flat index (exact search)
            index = faiss.IndexFlatL2(dimension)
            print(f"   Using IndexFlatL2 (exact search)")
            
        elif index_type == 'IVFFlat':
            # Inverted file index (approximate search, faster)
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            print(f"   Training IVFFlat index with {nlist} clusters...")
            index.train(embeddings)
            print(f"   Training complete")
            
            index.nprobe = 10  # Number of clusters to search
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add vectors to index
        print(f"\n📥 Adding vectors to index...")
        index.add(embeddings)
        
        print(f" FAISS index built successfully!")
        print(f"   Index contains: {index.ntotal:,} vectors")
        
        # Test the index
        self._test_faiss_index(index, embeddings)
        
        # Save the index
        self._save_faiss_index(index)
        
        return index
    
    def _test_faiss_index(self, index, embeddings, n_test=5):
        """
        Test FAISS index with sample queries
        """
        print(f"\n🧪 Testing FAISS index...")
        
        # Use first few chunks as test queries
        test_queries = embeddings[:n_test]
        test_texts = self.chunks[:n_test]
        
        for i in range(min(n_test, 3)):  # Test first 3 queries
            query = test_queries[i].reshape(1, -1).astype('float32')
            
            # Search for similar vectors
            k = 3  # Number of nearest neighbors
            distances, indices = index.search(query, k)
            
            print(f"\nQuery {i+1}: '{test_texts[i][:100]}...'")
            print(f"Top {k} results:")
            
            for j, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks):
                    result_text = self.chunks[idx]
                    metadata = self.metadata[idx]
                    print(f"  {j+1}. Distance: {dist:.4f}")
                    print(f"     Text: '{result_text[:100]}...'")
                    print(f"     Product: {metadata.get('product_category', 'Unknown')}")
                    print(f"     Issue: {metadata.get('issue', 'Unknown')}")
        
        print(f"\n FAISS index test completed successfully!")
    
    def _save_faiss_index(self, index, output_dir='vector_store/faiss_index'):
        """
        Save FAISS index and metadata
        
        Args:
            index: FAISS index
            output_dir: Directory to save index files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_dir / 'index.faiss'
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = output_dir / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save chunk texts
        chunks_path = output_dir / 'chunks.pkl'
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save configuration
        config = {
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'index_type': 'FAISS',
            'vector_dimension': index.d,
            'total_vectors': index.ntotal
        }
        
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n FAISS index saved to: {index_path}")
        print(f"   Metadata saved to: {metadata_path}")
        print(f"   Chunks saved to: {chunks_path}")
        print(f"   Configuration saved to: {config_path}")
        print(f"   Total size: {sum(p.stat().st_size for p in output_dir.glob('*')) / 1024**2:.2f} MB")
    
    def build_chroma_db(self, embeddings=None, chunks=None, metadata=None):
        """
        Build ChromaDB vector database
        
        Args:
            embeddings: Embedding vectors
            chunks: Text chunks
            metadata: Chunk metadata
            
        Returns:
            chromadb.Collection: ChromaDB collection
        """
        print(f"\n" + "="*60)
        print(f"  BUILDING CHROMADB DATABASE")
        print("="*60)
        
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings available. Run generate_embeddings() first.")
            embeddings = self.embeddings
        
        if chunks is None:
            if not self.chunks:
                raise ValueError("No chunks available. Run chunk_complaints() first.")
            chunks = self.chunks
        
        if metadata is None:
            if not self.metadata:
                raise ValueError("No metadata available. Run chunk_complaints() first.")
            metadata = self.metadata
        
        print(f" ChromaDB parameters:")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   Number of documents: {len(chunks):,}")
        print(f"   Collection name: complaint_chunks")
        
        # Initialize ChromaDB client
        chroma_dir = Path('vector_store/chroma_db')
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection(
            name="complaint_chunks",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"\n📥 Adding documents to ChromaDB...")
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Add in batches to avoid memory issues
        batch_size = 10000
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Adding to ChromaDB"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chunks))
            
            batch_ids = ids[start_idx:end_idx]
            batch_documents = chunks[start_idx:end_idx]
            batch_metadatas = metadata[start_idx:end_idx]
            batch_embeddings = embeddings[start_idx:end_idx].tolist()
            
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
        
        print(f"\n ChromaDB built successfully!")
        print(f"   Collection contains: {collection.count()} documents")
        
        # Test the collection
        self._test_chroma_collection(collection)
        
        # Save configuration
        self._save_chroma_config()
        
        return collection
    
    def _test_chroma_collection(self, collection, n_test=3):
        """
        Test ChromaDB collection with sample queries
        """
        print(f"\n🧪 Testing ChromaDB collection...")
        
        # Use first few chunks as test queries
        test_queries = self.chunks[:n_test]
        
        for i, query in enumerate(test_queries[:min(n_test, 3)]):
            print(f"\nQuery {i+1}: '{query[:100]}...'")
            
            results = collection.query(
                query_texts=[query],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"Top 3 results:")
            
            for j in range(len(results['documents'][0])):
                doc = results['documents'][0][j]
                meta = results['metadatas'][0][j]
                dist = results['distances'][0][j]
                
                print(f"  {j+1}. Distance: {dist:.4f}")
                print(f"     Text: '{doc[:100]}...'")
                print(f"     Product: {meta.get('product_category', 'Unknown')}")
                print(f"     Issue: {meta.get('issue', 'Unknown')}")
        
        print(f"\n✅ ChromaDB test completed successfully!")
    
    def _save_chroma_config(self):
        """
        Save ChromaDB configuration
        """
        config = {
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'database': 'ChromaDB',
            'collection_name': 'complaint_chunks',
            'similarity_metric': 'cosine',
            'total_documents': len(self.chunks)
        }
        
        config_path = Path('vector_store/chroma_db/config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 ChromaDB configuration saved to: {config_path}")
    
    def run_complete_pipeline(self, 
                            sample_size=12000, 
                            use_chroma=True,
                            save_embeddings=True):
        """
        Run complete vector store building pipeline
        
        Args:
            sample_size: Size of stratified sample
            use_chroma: If True, build ChromaDB; else build FAISS
            save_embeddings: If True, save embeddings to disk
            
        Returns:
            tuple: (vector_store, chunks, metadata, embeddings)
        """
        print("="*80)
        print("🚀 STARTING VECTOR STORE BUILDING PIPELINE")
        print("="*80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Create stratified sample
        self.stratified_sampling(sample_size=sample_size)
        
        # Step 3: Chunk complaints
        chunks, metadata = self.chunk_complaints()
        
        # Step 4: Generate embeddings
        embeddings = self.generate_embeddings()
        
        # Step 5: Build vector store
        if use_chroma:
            print(f"\n🔨 Building ChromaDB vector store...")
            vector_store = self.build_chroma_db(embeddings, chunks, metadata)
        else:
            print(f"\n🔨 Building FAISS vector store...")
            vector_store = self.build_faiss_index(embeddings)
        
        # Step 6: Save everything
        if save_embeddings:
            self._save_all_artifacts(chunks, metadata, embeddings, use_chroma)
        
        print("\n" + "="*80)
        print("🎉 VECTOR STORE PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return vector_store, chunks, metadata, embeddings
    
    def _save_all_artifacts(self, chunks, metadata, embeddings, use_chroma):
        """
        Save all pipeline artifacts
        """
        artifacts_dir = Path('data/processed/vector_store_artifacts')
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks as JSON
        chunks_path = artifacts_dir / 'chunks.json'
        with open(chunks_path, 'w', encoding='utf-8') as f:
            # Save sample of chunks
            sample_size = min(1000, len(chunks))
            sample_chunks = chunks[:sample_size]
            json.dump(sample_chunks, f, indent=2, ensure_ascii=False)
        
        # Save metadata sample
        metadata_path = artifacts_dir / 'metadata_sample.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            sample_metadata = metadata[:100]  # Save first 100 for inspection
            json.dump(sample_metadata, f, indent=2, ensure_ascii=False)
        
        # Save embeddings info
        embeddings_info = {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'total_size_mb': embeddings.nbytes / 1024**2,
            'model': self.embedding_model_name,
            'vector_store': 'ChromaDB' if use_chroma else 'FAISS'
        }
        
        info_path = artifacts_dir / 'embeddings_info.json'
        with open(info_path, 'w') as f:
            json.dump(embeddings_info, f, indent=2)
        
        print(f"\n💾 Pipeline artifacts saved to: {artifacts_dir}")
        print(f"   Chunks sample: {chunks_path}")
        print(f"   Metadata sample: {metadata_path}")
        print(f"   Embeddings info: {info_path}")


# Utility functions for working with pre-built vector stores
def load_prebuilt_vector_store(vector_store_path='vector_store/chroma_db'):
    """
    Load pre-built vector store
    
    Args:
        vector_store_path: Path to vector store directory
        
    Returns:
        Loaded vector store
    """
    vector_store_path = Path(vector_store_path)
    
    if not vector_store_path.exists():
        raise FileNotFoundError(f"Vector store not found at: {vector_store_path}")
    
    print(f"📂 Loading pre-built vector store from: {vector_store_path}")
    
    # Check if it's ChromaDB
    if (vector_store_path / 'chroma.sqlite3').exists():
        # Load ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=str(vector_store_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = chroma_client.get_collection("complaint_chunks")
        print(f"✅ ChromaDB loaded with {collection.count()} documents")
        return {'type': 'chroma', 'collection': collection}
    
    # Check if it's FAISS
    elif (vector_store_path / 'index.faiss').exists():
        # Load FAISS
        index = faiss.read_index(str(vector_store_path / 'index.faiss'))
        
        # Load metadata
        metadata_path = vector_store_path / 'metadata.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✅ FAISS index loaded with {index.ntotal} vectors")
        return {'type': 'faiss', 'index': index, 'metadata': metadata}
    
    else:
        raise ValueError(f"Unknown vector store format at: {vector_store_path}")


def test_semantic_search(vector_store, query, k=5):
    """
    Test semantic search on vector store
    
    Args:
        vector_store: Loaded vector store
        query: Search query
        k: Number of results
        
    Returns:
        List of search results
    """
    print(f"\n🔍 Testing semantic search with query: '{query}'")
    
    # Initialize embedding model (same as used for building)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode([query])[0]
    
    if vector_store['type'] == 'chroma':
        # ChromaDB search
        results = vector_store['collection'].query(
            query_texts=[query],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"📊 Found {len(results['documents'][0])} results:")
        
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            
            print(f"\n{i+1}. Similarity: {1-dist:.4f}")
            print(f"   Text: '{doc[:150]}...'")
            print(f"   Product: {meta.get('product_category', 'Unknown')}")
            print(f"   Issue: {meta.get('issue', 'Unknown')}")
            print(f"   Company: {meta.get('company', 'Unknown')}")
    
    else:  # FAISS
        # FAISS search
        distances, indices = vector_store['index'].search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k
        )
        
        print(f"📊 Found {len(indices[0])} results:")
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(vector_store['metadata']):
                metadata = vector_store['metadata'][idx]
                text = metadata.get('text', 'No text available')
                
                print(f"\n{i+1}. Distance: {dist:.4f}")
                print(f"   Text: '{text[:150]}...'")
                print(f"   Product: {metadata.get('product_category', 'Unknown')}")
                print(f"   Issue: {metadata.get('issue', 'Unknown')}")
                print(f"   Company: {metadata.get('company', 'Unknown')}")
    
    return results


# Main execution
if __name__ == "__main__":
    # Example: Build vector store from scratch
    builder = VectorStoreBuilder(
        data_path='data/processed/filtered_complaints.csv',
        embedding_model_name='all-MiniLM-L6-v2',
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Run complete pipeline (for learning purposes - uses 12K sample)
    print("🔨 Building vector store from sample data (for learning)...")
    vector_store, chunks, metadata, embeddings = builder.run_complete_pipeline(
        sample_size=12000,
        use_chroma=True,  # Change to False for FAISS
        save_embeddings=True
    
    )
    
    # Test with sample queries
    test_queries = [
        "credit card late fee",
        "loan payment issue",
        "bank account error"
    ]
    
    for query in test_queries:
        test_semantic_search(
            vector_store if isinstance(vector_store, dict) else {'type': 'chroma', 'collection': vector_store},
            query,
            k=3
        )
    
    # Example: Load pre-built vector store (for Tasks 3-4)
    print("\n" + "="*80)
    print(" LOADING PRE-BUILT VECTOR STORE (for Tasks 3-4)")
    print("="*80)
    
    try:
        # Try to load pre-built store
        prebuilt_store = load_prebuilt_vector_store('vector_store/chroma_db')
        
        # Test with more queries
        print("\n Testing pre-built vector store with business questions:")
        
        business_queries = [
            "What are common billing disputes with credit cards?",
            "How do customers complain about loan servicing?",
            "What issues do customers face with money transfers?"
        ]
        
        for query in business_queries:
            test_semantic_search(prebuilt_store, query, k=2)
            
    except FileNotFoundError:
        print("  Pre-built vector store not found. This is expected for Task 2.")
        print("   The sample vector store has been built and is ready for Task 3.")
        