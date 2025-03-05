import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Tuple

class HybridTransformer(nn.Module):
    def __init__(self, 
                 model_name: str, 
                 num_classes: int = 3, 
                 temperature: float = 0.05,
                 embedding_dim: int = 128,
                 shared_encoder: bool = False):
        """
        Initialize the Hybrid transformer that combines TLP and SimCSE approaches.
        
        Args:
            model_name: Name of the HuggingFace model to use
            num_classes: Number of bias classes (default: 3 for left/center/right)
            temperature: Temperature parameter for SimCSE contrastive loss scaling
            embedding_dim: Dimension of the projection space (default: 128)
            shared_encoder: Whether to share the encoder between TLP and SimCSE
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.temperature = temperature
        self.shared_encoder = shared_encoder
        
        # Separate dropout layers for each objective
        self.triplet_dropout = nn.Dropout(0.1)
        self.simcse_dropout = nn.Dropout(0.1)
        self.cl_dropout = nn.Dropout(0.2)  # Higher dropout for SimCSE augmentation
        self.classifier_dropout = nn.Dropout(0.1)
        
        # Projection head for triplet loss (bias-source separation)
        self.triplet_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim)
        )
        
        # Projection head for SimCSE (robust representations)
        self.simcse_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim)
        )
        
        # Bias classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               apply_cl_dropout: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input text into embeddings and logits.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            apply_cl_dropout: Whether to apply contrastive learning dropout
            
        Returns:
            Tuple of (triplet_embeddings, simcse_embeddings, logits)
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Get embeddings for triplet loss
        triplet_embeddings = self.triplet_projection(self.triplet_dropout(pooled_output))
        triplet_embeddings = F.normalize(triplet_embeddings, p=2, dim=1)
        
        # Get embeddings for SimCSE
        if apply_cl_dropout:
            simcse_embeddings = self.simcse_projection(self.cl_dropout(pooled_output))
        else:
            simcse_embeddings = self.simcse_projection(self.simcse_dropout(pooled_output))
        simcse_embeddings = F.normalize(simcse_embeddings, p=2, dim=1)
        
        # Get bias predictions
        logits = self.classifier(self.classifier_dropout(pooled_output))
        
        return triplet_embeddings, simcse_embeddings, logits
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                positive_ids: torch.Tensor = None, positive_mask: torch.Tensor = None,
                negative_ids: torch.Tensor = None, negative_masks: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Encode anchor with both objectives
        anchor_triplet, anchor_simcse, anchor_logits = self.encode(input_ids, attention_mask)
        
        # Get second SimCSE embedding with different dropout
        _, anchor_simcse2, _ = self.encode(input_ids, attention_mask, apply_cl_dropout=True)
        
        outputs = {
            'triplet_embeddings': anchor_triplet,
            'simcse_embeddings1': anchor_simcse,
            'simcse_embeddings2': anchor_simcse2,
            'logits': anchor_logits
        }
        
        # Process positive and negative examples if provided (for triplet loss)
        if positive_ids is not None and negative_ids is not None:
            pos_triplet, _, _ = self.encode(positive_ids, positive_mask)
            
            # Process each negative example separately and stack the results
            neg_triplets = []
            batch_size, num_negatives = negative_ids.size()[:2]
            
            # Reshape negative tensors to process them individually
            negative_ids = negative_ids.view(-1, negative_ids.size(-1))
            negative_masks = negative_masks.view(-1, negative_masks.size(-1))
            
            # Process all negative examples at once
            neg_triplet, _, _ = self.encode(negative_ids, negative_masks)
            neg_triplet = neg_triplet.view(batch_size, num_negatives, -1)
            
            # Use the first negative example for triplet loss
            outputs['positive_embeddings'] = pos_triplet
            outputs['negative_embeddings'] = neg_triplet[:, 0]
        
        return outputs