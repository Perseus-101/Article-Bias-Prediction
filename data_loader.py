import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional

class ArticleBiasDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 split_file: str,
                 tokenizer_name: str,
                 max_length: int = 512):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing the JSON article files
            split_file: Path to the split file (train.tsv, valid.tsv, or test.tsv)
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
        for article_id in self.df['ID']:  # Changed from 'article_id' to 'ID'
            with open(os.path.join(data_dir, 'jsons', f'{article_id}.json'), 'r') as f:
                self.articles[article_id] = json.load(f)
                
        # Create source to ID mapping
        sources = [self.articles[article_id]['source'] for article_id in self.df['ID']]  # Changed from 'article_id' to 'ID'
        self.source_to_id = {source: idx for idx, source in enumerate(set(sources))}
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        article = self.articles[row['ID']]  # Changed from 'article_id' to 'ID'
        
        # Tokenize the article content
        encoding = self.tokenizer(
            article['content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert labels
        bias_label = torch.tensor(article['bias'])
        source_label = torch.tensor(self.source_to_id[article['source']])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'bias_label': bias_label,
            'source_label': source_label
        }

def get_dataloaders(data_dir: str,
                   split_type: str,
                   tokenizer_name: str,
                   batch_size: int,
                   max_length: int = 512,
                   num_workers: int = 4,
                   pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation and test sets.
    
    Args:
        data_dir: Root directory containing the data
        split_type: Either 'media' or 'random'
        tokenizer_name: Name of the HuggingFace tokenizer to use
        batch_size: Batch size for the dataloaders
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    splits_dir = os.path.join(data_dir, 'splits', split_type)
    
    # Create datasets
    train_dataset = ArticleBiasDataset(
        data_dir=data_dir,
        split_file=os.path.join(splits_dir, 'train.tsv'),
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    
    val_dataset = ArticleBiasDataset(
        data_dir=data_dir,
        split_file=os.path.join(splits_dir, 'valid.tsv'),
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    
    test_dataset = ArticleBiasDataset(
        data_dir=data_dir,
        split_file=os.path.join(splits_dir, 'test.tsv'),
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader