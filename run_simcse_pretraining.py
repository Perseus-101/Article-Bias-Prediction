import os
import argparse
import torch
import pandas as pd
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from data_loader import get_dataloaders
from simcse_data_loader import get_simcse_dataloader
from simcse_model import SimCSETransformer
from simcse_trainer import SimCSETrainer
from trainer import BiasTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='SimCSE Pre-training for Political Bias Detection')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the dataset')
    parser.add_argument('--split_type', type=str, choices=['media', 'random'], default='random',
                        help='Type of dataset split to use')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='HuggingFace model name')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Temperature for scaling similarity scores')
    parser.add_argument('--supervised', action='store_true',
                        help='Use supervised SimCSE (ideology-based) instead of unsupervised')
    parser.add_argument('--use_fp16', action='store_true',
                        help='Use mixed precision training')
    
    # Pre-training arguments
    parser.add_argument('--pretrain_batch_size', type=int, default=32,
                        help='Batch size for pre-training')
    parser.add_argument('--pretrain_epochs', type=int, default=3,
                        help='Number of epochs for pre-training')
    parser.add_argument('--pretrain_lr', type=float, default=2e-5,
                        help='Learning rate for pre-training')
    parser.add_argument('--pretrain_grad_accum', type=int, default=2,
                        help='Gradient accumulation steps for pre-training')
    parser.add_argument('--patience', type=int, default=2,
                        help='Number of epochs to wait for improvement before early stopping')
    
    # Fine-tuning arguments
    parser.add_argument('--finetune_batch_size', type=int, default=32,
                        help='Batch size for fine-tuning')
    parser.add_argument('--finetune_epochs', type=int, default=3,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--finetune_lr', type=float, default=5e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--finetune_grad_accum', type=int, default=1,
                        help='Gradient accumulation steps for fine-tuning')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps for scheduler')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--pretrained_weights_path', type=str, default='./simcse_pretrained_weights.pt',
                        help='Path to save/load pre-trained weights')
    parser.add_argument('--skip_pretraining', action='store_true',
                        help='Skip pre-training and load weights from pretrained_weights_path')
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def format_results(results: Dict[str, float], model_name: str, split_type: str, supervised: bool) -> pd.DataFrame:
    """Format results as a DataFrame row."""
    simcse_type = "supervised" if supervised else "unsupervised"
    return pd.DataFrame({
        'Model': [f"{model_name}-{simcse_type}-simcse"],
        'Split': [split_type],
        'Macro F1': [f"{results['test_macro_f1']:.2f}"],
        'Acc.': [f"{results['test_accuracy']:.2f}"],
        'MAE': [f"{results['test_mae']:.2f}"]
    })

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device and precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.use_fp16 and torch.cuda.is_available():
        print("Using mixed precision training")
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Phase 1: Pre-training with SimCSE
    if not args.skip_pretraining:
        print(f"\n=== Phase 1: Pre-training with {('Supervised' if args.supervised else 'Unsupervised')} SimCSE ===")
        
        # Create SimCSE dataloader with optimized settings
        print(f"Loading SimCSE data from {args.split_type} split...")
        simcse_loader = get_simcse_dataloader(
            data_dir=args.data_dir,
            split_type=args.split_type,
            tokenizer_name=args.model_name,
            batch_size=args.pretrain_batch_size,
            max_length=args.max_length,
            supervised=args.supervised,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Create model for pre-training
        print(f"Creating SimCSETransformer model with {args.model_name}...")
        pretrain_model = SimCSETransformer(
            model_name=args.model_name,
            temperature=args.temperature,
            supervised=args.supervised
        )
        
        # Create pre-trainer
        pretrain_trainer = SimCSETrainer(
            model=pretrain_model,
            train_loader=simcse_loader,
            device=device,
            learning_rate=args.pretrain_lr,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.pretrain_grad_accum
        )
        
        # Pre-train model with early stopping
        print(f"Pre-training for {args.pretrain_epochs} epochs with early stopping (patience={args.patience})...")
        pretrain_metrics = pretrain_trainer.train(args.pretrain_epochs, patience=args.patience, scaler=scaler)
        
        # Save pre-trained weights
        pretrain_trainer.save_pretrained_model(args.pretrained_weights_path)
    
    # Phase 2: Fine-tuning for Political Bias Classification
    print("\n=== Phase 2: Fine-tuning for Political Bias Classification ===")
    
    # Get dataloaders for fine-tuning
    print(f"Loading {args.split_type} split data for fine-tuning...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        split_type=args.split_type,
        tokenizer_name=args.model_name,
        batch_size=args.finetune_batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model for fine-tuning
    print(f"Creating SimCSETransformer model for fine-tuning...")
    finetune_model = SimCSETransformer(
        model_name=args.model_name,
        temperature=args.temperature,
        supervised=args.supervised
    )
    
    # Load pre-trained weights if available
    if os.path.exists(args.pretrained_weights_path):
        print(f"Loading pre-trained weights from {args.pretrained_weights_path}")
        finetune_model = SimCSETrainer.load_pretrained_weights(
            model=finetune_model,
            weights_path=args.pretrained_weights_path
        )
    else:
        print("No pre-trained weights found. Starting fine-tuning from scratch.")
    
    # Create trainer for fine-tuning
    finetune_trainer = BiasTrainer(
        model=finetune_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.finetune_lr,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.finetune_grad_accum
    )
    
    # Fine-tune model with early stopping
    print(f"Fine-tuning for up to {args.finetune_epochs} epochs with early stopping (patience={args.patience})...")
    finetune_metrics = finetune_trainer.train(args.finetune_epochs, patience=args.patience)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = finetune_trainer.evaluate('test')
    print(f"Test metrics: {' '.join(f'{k}: {v:.2f}' for k, v in test_metrics.items())}")
    
    # Format and save results
    results_df = format_results(test_metrics, args.model_name, args.split_type, args.supervised)
    simcse_type = "supervised" if args.supervised else "unsupervised"
    results_path = os.path.join(args.output_dir, f"{args.model_name.replace('/', '-')}-{simcse_type}-simcse-{args.split_type}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Print results table
    print("\nResults:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()