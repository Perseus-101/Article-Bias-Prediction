import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from simcse_model import SimCSETransformer

class SimCSETrainer:
    def __init__(self,
                 model: SimCSETransformer,
                 train_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 0,
                 gradient_accumulation_steps: int = 1):
        """
        Initialize the SimCSE trainer.
        
        Args:
            model: The SimCSETransformer model to pre-train
            train_loader: DataLoader for SimCSE training data
            device: Device to train on (cuda/cpu)
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps for scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimizer and scheduler setup
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using SimCSE loss."""
        self.model.train()
        total_loss = 0
        total_cl_loss = 0
        total_ce_loss = 0
        steps = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            loss, cl_loss, ce_loss = self._training_step(batch)
            
            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            total_cl_loss += cl_loss.item()
            total_ce_loss += ce_loss.item()
            steps += 1
        
        return {
            'train_loss': total_loss / steps,
            'contrastive_loss': total_cl_loss / steps,
            'ce_loss': total_ce_loss / steps
        }
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single training step with SimCSE loss."""
        if self.model.supervised:
            return self._supervised_training_step(batch)
        else:
            return self._unsupervised_training_step(batch)
            
    def _train_epoch_mixed_precision(self, scaler: torch.cuda.amp.GradScaler) -> Dict[str, float]:
        """Train for one epoch using mixed precision."""
        self.model.train()
        total_loss = 0
        total_cl_loss = 0
        total_ce_loss = 0
        steps = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                loss, cl_loss, ce_loss = self._training_step(batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            total_cl_loss += cl_loss.item()
            total_ce_loss += ce_loss.item()
            steps += 1
        
        return {
            'train_loss': total_loss / steps,
            'contrastive_loss': total_cl_loss / steps,
            'ce_loss': total_ce_loss / steps
        }
    
    def _supervised_training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training step for supervised SimCSE."""
        # Move inputs to device
        anchor_input_ids = batch['anchor_input_ids'].to(self.device)
        anchor_attention_mask = batch['anchor_attention_mask'].to(self.device)
        positive_input_ids = batch['positive_input_ids'].to(self.device)
        positive_attention_mask = batch['positive_attention_mask'].to(self.device)
        negative_input_ids = batch['negative_input_ids'].to(self.device)
        negative_attention_masks = batch['negative_attention_masks'].to(self.device)
        anchor_bias = batch['anchor_bias'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=anchor_input_ids,
            attention_mask=anchor_attention_mask,
            positive_ids=positive_input_ids,
            positive_mask=positive_attention_mask,
            negative_ids=negative_input_ids,
            negative_masks=negative_attention_masks
        )
        
        # Calculate contrastive loss
        pos_sim = outputs['positive_sim']
        neg_sim = outputs['negative_sim']
        
        # For each anchor, we have one positive and multiple negatives
        labels = torch.zeros(pos_sim.size(0), device=self.device, dtype=torch.long)
        cl_loss = self.ce_loss(torch.cat([pos_sim, neg_sim], dim=1), labels)
        
        # Calculate classification loss
        ce_loss = self.ce_loss(outputs['logits'], anchor_bias)
        
        # Combine losses
        total_loss = cl_loss + ce_loss
        
        return total_loss, cl_loss, ce_loss
    
    def _unsupervised_training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training step for unsupervised SimCSE."""
        # Move inputs to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        bias_label = batch['bias_label'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Calculate contrastive loss with in-batch negatives
        sim_matrix = outputs['sim_matrix']
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=self.device)
        cl_loss = self.ce_loss(sim_matrix, labels)
        
        # Calculate classification loss
        ce_loss = self.ce_loss(outputs['logits'], bias_label)
        
        # Combine losses
        total_loss = cl_loss + ce_loss
        
        return total_loss, cl_loss, ce_loss
    
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
        # Save only the transformer and projection head weights
        # We don't save the classifier weights as they weren't trained
        state_dict = {
            'transformer': self.model.transformer.state_dict(),
            'projection': self.model.projection.state_dict()
        }
        torch.save(state_dict, save_path)
        print(f"Pre-trained model saved to {save_path}")
    
    @staticmethod
    def load_pretrained_weights(model: SimCSETransformer, weights_path: str):
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