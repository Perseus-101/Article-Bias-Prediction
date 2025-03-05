import os
import json
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional

class HybridArticleDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 split_file: str,
                 tokenizer_name: str,
                 max_length: int = 512):
        """
        Initialize the hybrid dataset that supports both TLP and SimCSE objectives.
        
        Args:
            data_dir: Directory containing the JSON article files
            split_file: Path to the split file (train.tsv)
            tokenizer_name: Name of the HuggingFace tokenizer to use
            max_length: Maximum sequence length for tokenization
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Read split file
        self.df = pd.read_csv(split_file, sep='\t')
        
        # Load all articles
        self.articles = {}
        for article_id in self.df['ID']:
            with open(os.path.join(data_dir, 'jsons', f'{article_id}.json'), 'r') as f:
                self.articles[article_id] = json.load(f)
        
        # Create mappings for bias and source
        self.bias_to_articles = {}
        self.source_to_articles = {}
        
        for article_id in self.df['ID']:
            article = self.articles[article_id]
            bias = article['bias']
            source = article['source']
            
            if bias not in self.bias_to_articles:
                self.bias_to_articles[bias] = []
            self.bias_to_articles[bias].append(article_id)
            
            if source not in self.source_to_articles:
                self.source_to_articles[source] = []
            self.source_to_articles[source].append(article_id)
        
        # Create hybrid triplets (anchor, positive, negative)
        self.triplets = self._create_hybrid_triplets()
    
    def _create_hybrid_triplets(self) -> List[Tuple[str, str, List[str]]]:
        """
        Create triplets that work for both TLP and SimCSE objectives.
        Each triplet contains:
        - Anchor: An article with a specific political bias
        - Positive: Another article with the same bias but from a different source
        - Negatives: Multiple articles with different bias (for SimCSE negatives)
          including at least one from the same source as anchor (for TLP negative)
        
        Returns:
            List of triplets (anchor_id, positive_id, [negative_ids])
        """
        triplets = []
        
        for article_id in self.df['ID']:
            article = self.articles[article_id]
            bias = article['bias']
            source = article['source']
            
            # Find positive examples (same bias, different source)
            same_bias_articles = [aid for aid in self.bias_to_articles[bias] 
                                if self.articles[aid]['source'] != source]
            
            # Find TLP negative examples (different bias, same source)
            same_source_diff_bias = [aid for aid in self.source_to_articles[source] 
                                   if self.articles[aid]['bias'] != bias]
            
            # Find additional SimCSE negative examples (different bias, any source)
            different_bias_articles = []
            for b in self.bias_to_articles:
                if b != bias:
                    different_bias_articles.extend(self.bias_to_articles[b])
            
            # Create triplets if we have both positive and negative examples
            if same_bias_articles and same_source_diff_bias:
                # Select one positive example
                positive_id = random.choice(same_bias_articles)
                
                # Ensure we have at least one TLP negative (same source, different bias)
                tlp_negative = random.choice(same_source_diff_bias)
                
                # Select additional negatives for SimCSE (up to 4 more)
                additional_negatives = []
                potential_negatives = [n for n in different_bias_articles if n != tlp_negative]
                if potential_negatives:
                    num_additional = min(4, len(potential_negatives))
                    additional_negatives = random.sample(potential_negatives, num_additional)
                
                # Combine TLP negative with additional SimCSE negatives
                all_negatives = [tlp_negative] + additional_negatives
                
                triplets.append((article_id, positive_id, all_negatives))
        
        print(f"Created {len(triplets)} hybrid triplets for training")
        return triplets
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_id, positive_id, negative_ids = self.triplets[idx]
        
        # Get articles
        anchor_article = self.articles[anchor_id]
        positive_article = self.articles[positive_id]
        
        # Tokenize anchor
        anchor_encoding = self.tokenizer(
            anchor_article['content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize positive
        positive_encoding = self.tokenizer(
            positive_article['content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize negatives
        negative_input_ids = []
        negative_attention_masks = []
        
        for neg_id in negative_ids:
            negative_article = self.articles[neg_id]
            negative_encoding = self.tokenizer(
                negative_article['content'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            negative_input_ids.append(negative_encoding['input_ids'].squeeze(0))
            negative_attention_masks.append(negative_encoding['attention_mask'].squeeze(0))
        
        # Stack negative tensors
        negative_input_ids = torch.stack(negative_input_ids)
        negative_attention_masks = torch.stack(negative_attention_masks)
        
        # Get bias label
        bias_label = torch.tensor(anchor_article['bias'])
        
        return {
            'input_ids': anchor_encoding['input_ids'].squeeze(0),
            'attention_mask': anchor_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_input_ids,
            'negative_attention_masks': negative_attention_masks,
            'bias_label': bias_label
        }

def get_hybrid_dataloader(data_dir: str,
                         split_type: str,
                         tokenizer_name: str,
                         batch_size: int,
                         max_length: int = 512,
                         num_workers: int = 0,
                         pin_memory: bool = False) -> DataLoader:
    """
    Create DataLoader for hybrid pre-training (TLP + SimCSE).
    
    Args:
        data_dir: Root directory containing the data
        split_type: Either 'media' or 'random'
        tokenizer_name: Name of the HuggingFace tokenizer to use
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length for tokenization
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader for hybrid training
    """
    splits_dir = os.path.join(data_dir, 'splits', split_type)
    
    # Create dataset using only training data
    hybrid_dataset = HybridArticleDataset(
        data_dir=data_dir,
        split_file=os.path.join(splits_dir, 'train.tsv'),
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    
    # Create dataloader
    hybrid_loader = DataLoader(
        hybrid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return hybrid_loader