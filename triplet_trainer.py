import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from models import TripletTransformer

class TripletPreTrainer:
    def __init__(self,
                 model: TripletTransformer,
                 triplet_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 0,
                 gradient_accumulation_steps: int = 1):
        """
        Initialize the triplet pre-trainer.
        
        Args:
            model: The TripletTransformer model to pre-train
            triplet_loader: DataLoader for triplet training data
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps for scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model.to(device)
        self.triplet_loader = triplet_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimizer and scheduler setup
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(triplet_loader) * gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Triplet loss function with margin=1.0 as specified in the paper
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using triplet loss."""
        self.model.train()
        total_loss = 0
        steps = 0
        
        for batch_idx, batch in enumerate(tqdm(self.triplet_loader)):
            loss = self._training_step(batch)
            
            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            steps += 1
        
        return {'train_loss': total_loss / steps}
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step with triplet loss."""
        # Process anchor
        anchor_input_ids = batch['anchor_input_ids'].to(self.device)
        anchor_attention_mask = batch['anchor_attention_mask'].to(self.device)
        anchor_outputs = self.model(input_ids=anchor_input_ids, attention_mask=anchor_attention_mask)
        anchor_embeddings = anchor_outputs['embeddings']
        
        # Process positive
        positive_input_ids = batch['positive_input_ids'].to(self.device)
        positive_attention_mask = batch['positive_attention_mask'].to(self.device)
        positive_outputs = self.model(input_ids=positive_input_ids, attention_mask=positive_attention_mask)
        positive_embeddings = positive_outputs['embeddings']
        
        # Process negative
        negative_input_ids = batch['negative_input_ids'].to(self.device)
        negative_attention_mask = batch['negative_attention_mask'].to(self.device)
        negative_outputs = self.model(input_ids=negative_input_ids, attention_mask=negative_attention_mask)
        negative_embeddings = negative_outputs['embeddings']
        
        # Calculate triplet loss
        loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        return loss
    
    def train(self, num_epochs: int) -> List[Dict[str, float]]:
        """Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            
        Returns:
            List of metric dictionaries for each epoch
        """
        metrics_history = []
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Training
            train_metrics = self.train_epoch()
            metrics_history.append(train_metrics)
            
            # Print metrics
            metrics_str = ' '.join(f'{k}: {v:.4f}' for k, v in train_metrics.items())
            print(f'Epoch {epoch + 1} metrics: {metrics_str}')
        
        return metrics_history
    
    def save_pretrained_model(self, save_path: str):
        """Save the pre-trained model weights.
        
        Args:
            save_path: Path to save the model
        """
        # Save only the transformer and projection head weights
        # We don't save the classifier weights as they weren't trained
        state_dict = {
            'transformer': self.model.transformer.state_dict(),
            'projection': self.model.projection.state_dict()
        }
        torch.save(state_dict, save_path)
        print(f"Pre-trained model saved to {save_path}")
    
    @staticmethod
    def load_pretrained_weights(model: TripletTransformer, weights_path: str):
        """Load pre-trained weights into a model.
        
        Args:
            model: The model to load weights into
            weights_path: Path to the saved weights
        
        Returns:
            Model with loaded weights
        """
        state_dict = torch.load(weights_path)
        model.transformer.load_state_dict(state_dict['transformer'])
        model.projection.load_state_dict(state_dict['projection'])
        return model