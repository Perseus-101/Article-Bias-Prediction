import os
import argparse
import torch
import pandas as pd
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from data_loader import get_dataloaders
from models import BaselineTransformer, AdversarialMediaTransformer, TripletTransformer
from trainer import BiasTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Political Bias Detection')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the dataset')
    parser.add_argument('--split_type', type=str, choices=['media', 'random'], default='random',  # Changed to random for faster initial testing
                        help='Type of dataset split to use')
    parser.add_argument('--max_length', type=int, default=128,  # Reduced for faster testing
                        help='Maximum sequence length for tokenization')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['baseline', 'adversarial', 'triplet'],
                        default='baseline', help='Type of model to use')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',  # Changed to a smaller model
                        help='HuggingFace model name')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,  # Increased for faster training
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,  # Reduced for faster training
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--learning_rate', type=float, default=5e-5,  # Slightly increased for faster convergence
                        help='Learning rate for optimization')
    parser.add_argument('--num_epochs', type=int, default=1,  # Reduced for initial testing
                        help='Number of epochs to train for')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps for scheduler')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_model(args, num_sources: int):
    """Create model based on arguments."""
    if args.model_type == 'baseline':
        return BaselineTransformer(args.model_name)
    elif args.model_type == 'adversarial':
        return AdversarialMediaTransformer(args.model_name, num_source_classes=num_sources)
    elif args.model_type == 'triplet':
        return TripletTransformer(args.model_name)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

def format_results(results: Dict[str, float], model_name: str, split_type: str) -> pd.DataFrame:
    """Format results as a DataFrame row."""
    return pd.DataFrame({
        'Model': [model_name],
        'Split': [split_type],
        'Macro F1': [f"{results['test_macro_f1']:.2f}"],
        'Acc.': [f"{results['test_accuracy']:.2f}"],
        'MAE': [f"{results['test_mae']:.2f}"]
    })

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataloaders
    print(f"Loading {args.split_type} split data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        split_type=args.split_type,
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Get number of sources for adversarial model
    num_sources = len(train_loader.dataset.source_to_id)
    print(f"Number of media sources: {num_sources}")
    
    # Create model
    print(f"Creating {args.model_type} model with {args.model_name}...")
    model = create_model(args, num_sources)
    
    # Create trainer
    trainer = BiasTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Train model
    print(f"Training for {args.num_epochs} epochs...")
    metrics_history = trainer.train(args.num_epochs)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate('test')
    print(f"Test metrics: {' '.join(f'{k}: {v:.2f}' for k, v in test_metrics.items())}")
    
    # Format and save results
    results_df = format_results(test_metrics, f"{args.model_name}-{args.model_type}", args.split_type)
    results_path = os.path.join(args.output_dir, f"{args.model_name.replace('/', '-')}-{args.model_type}-{args.split_type}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Print results table
    print("\nResults:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()