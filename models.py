import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Tuple

class BaselineTransformer(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 3):
        """Initialize the baseline transformer classifier.
        
        Args:
            model_name: Name of the HuggingFace model to use
            num_classes: Number of bias classes (default: 3 for left/center/right)
        """
        super().__init__()
        # Initialize the transformer model without xformers to avoid compatibility issues
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation (first token) instead of pooler_output
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class AdversarialMediaTransformer(nn.Module):
    def __init__(self, model_name: str, num_bias_classes: int = 3, num_source_classes: int = 0):
        """Initialize the adversarial media-aware transformer.
        
        Args:
            model_name: Name of the HuggingFace model to use
            num_bias_classes: Number of bias classes (default: 3)
            num_source_classes: Number of media source classes
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()  # Enable gradient checkpointing if supported
        hidden_size = self.transformer.config.hidden_size
        
        # Bias classifier
        self.dropout = nn.Dropout(0.1)
        self.bias_classifier = nn.Linear(hidden_size, num_bias_classes)
        
        # Source classifier (adversarial)
        self.source_classifier = nn.Linear(hidden_size, num_source_classes)
        self.gradient_reversal_lambda = 1.0
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation (first token) instead of pooler_output
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # Bias prediction
        bias_logits = self.bias_classifier(pooled_output)
        
        # Source prediction with gradient reversal
        reversed_features = GradientReversal.apply(pooled_output, self.gradient_reversal_lambda)
        source_logits = self.source_classifier(reversed_features)
        
        return bias_logits, source_logits

class TripletTransformer(nn.Module):
    def __init__(self, model_name: str, num_bias_classes: int = 3):
        """Initialize the triplet loss transformer.
        
        Args:
            model_name: Name of the HuggingFace model to use
            num_bias_classes: Number of bias classes (default: 3)
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()  # Enable gradient checkpointing if supported
        hidden_size = self.transformer.config.hidden_size
        
        # Projection head for triplet loss
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)  # Embedding dimension
        )
        
        # Bias classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_bias_classes)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation (first token) instead of pooler_output
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Get embeddings for triplet loss
        embeddings = self.projection(pooled_output)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Get bias predictions
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return {
            'embeddings': embeddings,
            'logits': logits
        }

# Gradient Reversal Layer for adversarial training
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None