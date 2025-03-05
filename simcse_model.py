import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Tuple

class SimCSETransformer(nn.Module):
    def __init__(self, 
                 model_name: str, 
                 num_classes: int = 3, 
                 temperature: float = 0.05,
                 supervised: bool = True):
        """
        Initialize the SimCSE transformer for contrastive learning.
        
        Args:
            model_name: Name of the HuggingFace model to use
            num_classes: Number of bias classes (default: 3 for left/center/right)
            temperature: Temperature parameter for contrastive loss scaling
            supervised: Whether to use supervised or unsupervised SimCSE
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.temperature = temperature
        self.supervised = supervised
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)  # Embedding dimension
        )
        
        # Bias classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Additional dropout layer for unsupervised SimCSE
        if not supervised:
            self.cl_dropout = nn.Dropout(0.1)
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
               apply_cl_dropout: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input text into embeddings and logits.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            apply_cl_dropout: Whether to apply contrastive learning dropout
            
        Returns:
            Tuple of (embeddings, logits)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Get embeddings for contrastive loss
        if not self.supervised and apply_cl_dropout:
            embeddings = self.projection(self.cl_dropout(pooled_output))
        else:
            embeddings = self.projection(pooled_output)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Get bias predictions
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return embeddings, logits
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                positive_ids: torch.Tensor = None, positive_mask: torch.Tensor = None,
                negative_ids: torch.Tensor = None, negative_masks: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive learning.
        
        For supervised SimCSE:
            - Takes anchor, positive, and negative examples
            - Computes similarity between anchor-positive and anchor-negative pairs
        
        For unsupervised SimCSE:
            - Takes single input
            - Creates two views with different dropout masks
            - Uses in-batch negatives
        
        Args:
            input_ids: Input token IDs for anchor
            attention_mask: Attention mask for anchor
            positive_ids: Input token IDs for positive example (supervised only)
            positive_mask: Attention mask for positive example (supervised only)
            negative_ids: Input token IDs for negative examples (supervised only)
            negative_masks: Attention masks for negative examples (supervised only)
            
        Returns:
            Dictionary containing embeddings, logits, and similarity scores
        """
        if self.supervised:
            # Encode anchor, positive, and negative examples
            anchor_embeddings, anchor_logits = self.encode(input_ids, attention_mask)
            positive_embeddings, _ = self.encode(positive_ids, positive_mask)
            
            # Handle multiple negatives
            batch_size, num_negatives = negative_ids.shape[:2]
            negative_ids = negative_ids.view(-1, negative_ids.size(-1))
            negative_masks = negative_masks.view(-1, negative_masks.size(-1))
            negative_embeddings, _ = self.encode(negative_ids, negative_masks)
            negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)
            
            # Compute similarity scores
            pos_sim = torch.sum(anchor_embeddings.unsqueeze(1) * positive_embeddings.unsqueeze(1), dim=-1)
            neg_sim = torch.sum(anchor_embeddings.unsqueeze(1).unsqueeze(1) * negative_embeddings, dim=-1)
            
            # Scale by temperature
            pos_sim = pos_sim / self.temperature
            neg_sim = neg_sim / self.temperature
            
            return {
                'embeddings': anchor_embeddings,
                'logits': anchor_logits,
                'positive_sim': pos_sim,
                'negative_sim': neg_sim
            }
        
        else:  # Unsupervised SimCSE
            # Create two views of the same input with different dropout masks
            embeddings1, logits = self.encode(input_ids, attention_mask)
            embeddings2, _ = self.encode(input_ids, attention_mask, apply_cl_dropout=True)
            
            # Compute similarity matrix for all pairs in batch
            sim_matrix = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
            
            return {
                'embeddings': embeddings1,
                'logits': logits,
                'sim_matrix': sim_matrix
            }