import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from hybrid_model import HybridTransformer

class HybridTrainer:
    def __init__(self,
                 model: HybridTransformer,
                 train_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 0,
                 gradient_accumulation_steps: int = 1,
                 triplet_weight: float = 1.0,
                 simcse_weight: float = 1.0,
                 ce_weight: float = 1.0,
                 dynamic_weight: bool = True):
        """
        Initialize the hybrid trainer that combines TLP and SimCSE objectives.
        
        Args:
            model: The HybridTransformer model to pre-train
            train_loader: DataLoader for hybrid training data
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps for scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            triplet_weight: Weight for triplet loss component
            simcse_weight: Weight for SimCSE loss component
            ce_weight: Weight for classification loss component
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Loss weights and dynamic weighting
        self.triplet_weight = triplet_weight
        self.simcse_weight = simcse_weight
        self.ce_weight = ce_weight
        self.dynamic_weight = dynamic_weight
        self.loss_history = {'triplet': [], 'simcse': [], 'ce': []}
        self.weight_history = {'triplet': [], 'simcse': [], 'ce': []}
        
        # Optimizer and scheduler setup
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss functions
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using combined loss."""
        self.model.train()
        total_loss = 0
        total_triplet_loss = 0
        total_simcse_loss = 0
        total_ce_loss = 0
        steps = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            loss, triplet_loss, simcse_loss, ce_loss = self._training_step(batch)
            
            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            total_triplet_loss += triplet_loss.item()
            total_simcse_loss += simcse_loss.item()
            total_ce_loss += ce_loss.item()
            steps += 1
        
        return {
            'train_loss': total_loss / steps,
            'triplet_loss': total_triplet_loss / steps,
            'simcse_loss': total_simcse_loss / steps,
            'ce_loss': total_ce_loss / steps
        }
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single training step with combined loss."""
        # Move inputs to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        positive_ids = batch['positive_input_ids'].to(self.device)
        positive_mask = batch['positive_attention_mask'].to(self.device)
        negative_ids = batch['negative_input_ids'].to(self.device)
        negative_masks = batch['negative_attention_masks'].to(self.device)
        bias_label = batch['bias_label'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positive_ids=positive_ids,
            positive_mask=positive_mask,
            negative_ids=negative_ids,
            negative_masks=negative_masks
        )
        
        # Calculate individual losses
        triplet_loss = self.triplet_loss(
            outputs['triplet_embeddings'],
            outputs['positive_embeddings'],
            outputs['negative_embeddings']
        )
        
        # Calculate SimCSE loss
        sim_matrix = torch.matmul(outputs['simcse_embeddings1'], outputs['simcse_embeddings2'].t()) / self.model.temperature
        labels = torch.arange(sim_matrix.size(0), device=self.device)
        simcse_loss = self.ce_loss(sim_matrix, labels)
        
        # Calculate classification loss
        ce_loss = self.ce_loss(outputs['logits'], bias_label)
        
        # Update loss history
        self.loss_history['triplet'].append(triplet_loss.item())
        self.loss_history['simcse'].append(simcse_loss.item())
        self.loss_history['ce'].append(ce_loss.item())
        
        # Dynamic weight adjustment based on loss magnitudes and gradients
        if self.dynamic_weight and len(self.loss_history['triplet']) > 1:
            triplet_ratio = self.loss_history['triplet'][-1] / self.loss_history['triplet'][-2]
            simcse_ratio = self.loss_history['simcse'][-1] / self.loss_history['simcse'][-2]
            ce_ratio = self.loss_history['ce'][-1] / self.loss_history['ce'][-2]
            
            # Adjust weights based on loss ratios
            self.triplet_weight *= max(0.5, min(2.0, 1.0 / triplet_ratio))
            self.simcse_weight *= max(0.5, min(2.0, 1.0 / simcse_ratio))
            self.ce_weight *= max(0.5, min(2.0, 1.0 / ce_ratio))
            
            # Normalize weights
            total_weight = self.triplet_weight + self.simcse_weight + self.ce_weight
            self.triplet_weight /= total_weight
            self.simcse_weight /= total_weight
            self.ce_weight /= total_weight
        
        # Record current weights
        self.weight_history['triplet'].append(self.triplet_weight)
        self.weight_history['simcse'].append(self.simcse_weight)
        self.weight_history['ce'].append(self.ce_weight)
        
        # Combine losses with current weights
        total_loss = (
            self.triplet_weight * triplet_loss +
            self.simcse_weight * simcse_loss +
            self.ce_weight * ce_loss
        )
        
        return total_loss, triplet_loss, simcse_loss, ce_loss
    
    def _train_epoch_mixed_precision(self, scaler: torch.cuda.amp.GradScaler) -> Dict[str, float]:
        """Train for one epoch using mixed precision."""
        self.model.train()
        total_loss = 0
        total_triplet_loss = 0
        total_simcse_loss = 0
        total_ce_loss = 0
        steps = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                loss, triplet_loss, simcse_loss, ce_loss = self._training_step(batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            total_triplet_loss += triplet_loss.item()
            total_simcse_loss += simcse_loss.item()
            total_ce_loss += ce_loss.item()
            steps += 1
        
        return {
            'train_loss': total_loss / steps,
            'triplet_loss': total_triplet_loss / steps,
            'simcse_loss': total_simcse_loss / steps,
            'ce_loss': total_ce_loss / steps
        }
    
    def train(self, num_epochs: int, patience: int = None, scaler: torch.cuda.amp.GradScaler = None) -> List[Dict[str, float]]:
        """Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            patience: Number of epochs to wait for improvement before early stopping
            scaler: Optional GradScaler for mixed precision training
            
        Returns:
            List of metric dictionaries for each epoch
        """
        metrics_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Training
            if scaler is not None:
                train_metrics = self._train_epoch_mixed_precision(scaler)
            else:
                train_metrics = self.train_epoch()
            
            metrics_history.append(train_metrics)
            
            # Print metrics
            metrics_str = ' '.join(f'{k}: {v:.4f}' for k, v in train_metrics.items())
            print(f'Epoch {epoch + 1} metrics: {metrics_str}')
            
            # Early stopping check
            if patience is not None:
                current_loss = train_metrics['train_loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping triggered after {epoch + 1} epochs')
                        break
        
        return metrics_history
    
    def save_pretrained_model(self, save_path: str):
        """Save the pre-trained model weights.
        
        Args:
            save_path: Path to save the model
        """
        state_dict = {
            'transformer': self.model.transformer.state_dict(),
            'triplet_projection': self.model.triplet_projection.state_dict(),
            'simcse_projection': self.model.simcse_projection.state_dict()
        }
        torch.save(state_dict, save_path)
        print(f"Pre-trained model saved to {save_path}")
    
    @staticmethod
    def load_pretrained_weights(model: HybridTransformer, weights_path: str):
        """Load pre-trained weights into a model.
        
        Args:
            model: The model to load weights into
            weights_path: Path to the saved weights
        
        Returns:
            Model with loaded weights
        """
        state_dict = torch.load(weights_path)
        model.transformer.load_state_dict(state_dict['transformer'])
        model.triplet_projection.load_state_dict(state_dict['triplet_projection'])
        model.simcse_projection.load_state_dict(state_dict['simcse_projection'])
        return model