import os
import json
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional

class SimCSEArticleDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 split_file: str,
                 tokenizer_name: str,
                 max_length: int = 512,
                 supervised: bool = True):
        """
        Initialize the SimCSE dataset for contrastive pre-training.
        
        Args:
            data_dir: Directory containing the JSON article files
            split_file: Path to the split file (train.tsv)
            tokenizer_name: Name of the HuggingFace tokenizer to use
            max_length: Maximum sequence length for tokenization
            supervised: Whether to use supervised (ideology-based) or unsupervised (dropout-based) SimCSE
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.supervised = supervised
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
        
        # Create pairs for supervised SimCSE
        if supervised:
            self.pairs = self._create_supervised_pairs()
        else:
            # For unsupervised SimCSE, we just use the articles directly
            # and create augmentations on-the-fly with different dropout masks
            self.article_ids = list(self.df['ID'])
    
    def _create_supervised_pairs(self) -> List[Tuple[str, str, List[str]]]:
        """
        Create pairs for supervised SimCSE training.
        Each pair contains:
        - Anchor: An article with a specific political bias
        - Positive: Another article with the same bias but from a different source
        - Negatives: Articles with different bias (can be from any source)
        
        Returns:
            List of pairs (anchor_id, positive_id, [negative_ids])
        """
        pairs = []
        
        for article_id in self.df['ID']:
            article = self.articles[article_id]
            bias = article['bias']
            source = article['source']
            
            # Find positive examples (same bias, different source)
            same_bias_articles = [aid for aid in self.bias_to_articles[bias] 
                                if self.articles[aid]['source'] != source]
            
            # Find negative examples (different bias)
            different_bias_articles = []
            for b in self.bias_to_articles:
                if b != bias:
                    different_bias_articles.extend(self.bias_to_articles[b])
            
            # Create pairs if we have both positive and negative examples
            if same_bias_articles and different_bias_articles:
                # Select one positive example
                positive_id = random.choice(same_bias_articles)
                
                # Select multiple negative examples (up to 5)
                num_negatives = min(5, len(different_bias_articles))
                negative_ids = random.sample(different_bias_articles, num_negatives)
                
                pairs.append((article_id, positive_id, negative_ids))
        
        print(f"Created {len(pairs)} pairs for supervised SimCSE training")
        return pairs
    
    def __len__(self) -> int:
        if self.supervised:
            return len(self.pairs)
        else:
            return len(self.article_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.supervised:
            return self._get_supervised_item(idx)
        else:
            return self._get_unsupervised_item(idx)
    
    def _get_supervised_item(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_id, positive_id, negative_ids = self.pairs[idx]
        
        # Get articles
        anchor_article = self.articles[anchor_id]
        positive_article = self.articles[positive_id]
        
        # Tokenize anchor and positive
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
        
        # Get labels
        anchor_bias = torch.tensor(anchor_article['bias'])
        positive_bias = torch.tensor(positive_article['bias'])
        
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_input_ids,
            'negative_attention_masks': negative_attention_masks,
            'anchor_bias': anchor_bias,
            'positive_bias': positive_bias
        }
    
    def _get_unsupervised_item(self, idx: int) -> Dict[str, torch.Tensor]:
        article_id = self.article_ids[idx]
        article = self.articles[article_id]
        
        # Tokenize the article content
        encoding = self.tokenizer(
            article['content'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For unsupervised SimCSE, we return the same input twice
        # Different dropout masks will be applied during forward pass
        bias_label = torch.tensor(article['bias'])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'bias_label': bias_label
        }

def get_simcse_dataloader(data_dir: str,
                         split_type: str,
                         tokenizer_name: str,
                         batch_size: int,
                         max_length: int = 512,
                         supervised: bool = True,
                         num_workers: int = 0,
                         pin_memory: bool = False) -> DataLoader:
    """
    Create DataLoader for SimCSE pre-training.
    
    Args:
        data_dir: Root directory containing the data
        split_type: Either 'media' or 'random'
        tokenizer_name: Name of the HuggingFace tokenizer to use
        batch_size: Batch size for the dataloader
        max_length: Maximum sequence length for tokenization
        supervised: Whether to use supervised (ideology-based) or unsupervised (dropout-based) SimCSE
        
    Returns:
        DataLoader for SimCSE training
    """
    splits_dir = os.path.join(data_dir, 'splits', split_type)
    
    # Create dataset using only training data
    simcse_dataset = SimCSEArticleDataset(
        data_dir=data_dir,
        split_file=os.path.join(splits_dir, 'train.tsv'),
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        supervised=supervised
    )
    
    # Create dataloader
    simcse_loader = DataLoader(
        simcse_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return simcse_loader