import os
import json
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional

class TripletArticleDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 split_file: str,
                 tokenizer_name: str,
                 max_length: int = 512):
        """
        Initialize the triplet dataset for pre-training.
        
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
        
        # Create triplets
        self.triplets = self._create_triplets()
    
    def _create_triplets(self) -> List[Tuple[str, str, str]]:
        """
        Create triplets for training with triplet loss.
        Each triplet contains:
        - Anchor: An article with a specific political bias
        - Positive: Another article with the same bias but from a different source
        - Negative: An article with a different bias but from the same source as the anchor
        
        Returns:
            List of triplets (anchor_id, positive_id, negative_id)
        """
        triplets = []
        
        for article_id in self.df['ID']:
            article = self.articles[article_id]
            bias = article['bias']
            source = article['source']
            
            # Find positive examples (same bias, different source)
            same_bias_articles = [aid for aid in self.bias_to_articles[bias] 
                                if self.articles[aid]['source'] != source]
            
            # Find negative examples (different bias, same source)
            same_source_articles = [aid for aid in self.source_to_articles[source] 
                                  if self.articles[aid]['bias'] != bias]
            
            # Create triplets if we have both positive and negative examples
            if same_bias_articles and same_source_articles:
                for _ in range(min(3, len(same_bias_articles), len(same_source_articles))):
                    positive_id = random.choice(same_bias_articles)
                    negative_id = random.choice(same_source_articles)
                    triplets.append((article_id, positive_id, negative_id))
        
        print(f"Created {len(triplets)} triplets for training")
        return triplets
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_id, positive_id, negative_id = self.triplets[idx]
        
        # Get articles
        anchor_article = self.articles[anchor_id]
        positive_article = self.articles[positive_id]
        negative_article = self.articles[negative_id]
        
        # Tokenize articles
        anchor_encoding = self.tokenizer(
            anchor_article['content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            positive_article['content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        negative_encoding = self.tokenizer(
            negative_article['content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels
        anchor_bias = torch.tensor(anchor_article['bias'])
        positive_bias = torch.tensor(positive_article['bias'])
        negative_bias = torch.tensor(negative_article['bias'])
        
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0),
            'anchor_bias': anchor_bias,
            'positive_bias': positive_bias,
            'negative_bias': negative_bias
        }

def get_triplet_dataloader(data_dir: str,
                          split_type: str,
                          tokenizer_name: str,
                          batch_size: int,
                          max_length: int = 512) -> DataLoader:
    """
    Create DataLoader for triplet pre-training.
    
    Args:
        data_dir: Root directory containing the data
        split_type: Either 'media' or 'random'
        tokenizer_name: Name of the HuggingFace tokenizer to use
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length for tokenization
        
    Returns:
        DataLoader for triplet training
    """
    splits_dir = os.path.join(data_dir, 'splits', split_type)
    
    # Create dataset using only training data
    triplet_dataset = TripletArticleDataset(
        data_dir=data_dir,
        split_file=os.path.join(splits_dir, 'train.tsv'),
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    
    # Create dataloader
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    return triplet_loader