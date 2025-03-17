import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import numpy as np
from tqdm import tqdm
from models import TripletTransformer, AdversarialMediaTransformer

class BiasTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 0,
                 gradient_accumulation_steps: int = 1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize mixed precision scaler
        self.scaler = torch.amp.GradScaler()
        
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
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        steps = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            # Use autocast for mixed precision training
            with torch.amp.autocast(device_type=self.device.type):
                loss = self._training_step(batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Scale loss and compute gradients
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Unscale gradients and update parameters
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            steps += 1
        
        return {'train_loss': total_loss / steps}
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        bias_labels = batch['bias_label'].to(self.device)
        source_labels = batch['source_label'].to(self.device)
        
        if isinstance(self.model, TripletTransformer):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            embeddings = outputs['embeddings']
            
            triplet_loss = self._compute_batch_triplet_loss(embeddings, bias_labels)
            ce_loss = self.ce_loss(logits, bias_labels)
            loss = ce_loss + triplet_loss
            
        elif isinstance(self.model, AdversarialMediaTransformer):
            bias_logits, source_logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            bias_loss = self.ce_loss(bias_logits, bias_labels)
            source_loss = self.ce_loss(source_logits, source_labels)
            loss = bias_loss - 0.1 * source_loss
            
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            loss = self.ce_loss(logits, bias_labels)
        
        return loss
    
    def _compute_batch_triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss for a batch of embeddings."""
        triplet_loss = torch.tensor(0.0).to(self.device)
        num_triplets = 0
        
        for anchor_idx in range(len(embeddings)):
            anchor_label = labels[anchor_idx]
            anchor_embed = embeddings[anchor_idx]
            
            # Find positive and negative examples
            pos_mask = (labels == anchor_label) & (torch.arange(len(embeddings)).to(self.device) != anchor_idx)
            neg_mask = labels != anchor_label
            
            if not (pos_mask.any() and neg_mask.any()):
                continue
                
            positive = embeddings[pos_mask][0]  # Take first positive
            negative = embeddings[neg_mask][0]  # Take first negative
            
            triplet_loss += self.triplet_loss(anchor_embed.unsqueeze(0),
                                              positive.unsqueeze(0),
                                              negative.unsqueeze(0))
            num_triplets += 1
        
        return triplet_loss / max(num_triplets, 1)
    
    def evaluate(self, split: str = 'val') -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            split: One of 'val' or 'test'
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        loader = self.val_loader if split == 'val' else self.test_loader
        
        all_preds = []
        all_labels = []
        total_loss = 0
        steps = 0
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bias_labels = batch['bias_label'].to(self.device)
                
                if isinstance(self.model, AdversarialMediaTransformer):
                    logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                elif isinstance(self.model, TripletTransformer):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                else:  # Baseline transformer or SimCSE
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                
                loss = self.ce_loss(logits, bias_labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(bias_labels.cpu().numpy())
                steps += 1
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return {
            f'{split}_loss': total_loss / steps,
            f'{split}_macro_f1': f1_score(all_labels, all_preds, average='macro') * 100,
            f'{split}_accuracy': accuracy_score(all_labels, all_preds) * 100,
            f'{split}_mae': mean_absolute_error(all_labels, all_preds)
        }
    
    def train(self, num_epochs: int, patience: int = 2) -> List[Dict[str, float]]:
        """Train the model for specified number of epochs with early stopping.
        
        Args:
            num_epochs: Number of epochs to train for
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            List of metric dictionaries for each epoch
        """
        metrics_history = []
        best_val_metric = float('inf')  # Lower is better for loss
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.evaluate('val')
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            metrics_history.append(epoch_metrics)
            
            # Print metrics
            metrics_str = ' '.join(f'{k}: {v:.2f}' for k, v in epoch_metrics.items())
            print(f'Epoch {epoch + 1} metrics: {metrics_str}')
            
            # Early stopping check
            current_val_metric = val_metrics['val_loss']
            if current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break
        
        return metrics_history