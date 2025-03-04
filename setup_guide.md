# Political Bias Detection System - Setup Guide

This guide will help you set up and run the political bias detection system in a virtual environment, adjust parameters, and use third-party language models.

## 1. Setting Up a Virtual Environment

You can choose either Conda or venv for your virtual environment:

### Option A: Using Conda

```bash
# Create a new conda environment
conda create -n bias-detection python=3.9

# Activate the environment
conda activate bias-detection

# Install PyTorch with CUDA support (if you have a compatible GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Option B: Using venv (Python's built-in virtual environment)

```bash
# Create a new virtual environment
python -m venv bias-env

# Activate the environment
# On Windows:
bias-env\Scripts\activate
# On macOS/Linux:
source bias-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Running the Code

### Basic Run (Quick Test)

To quickly test if everything is working, run:

```bash
python main.py --num_epochs 1 --model_name distilbert-base-uncased --batch_size 16 --max_length 128
```

This will run a quick test with minimal settings:
- 1 epoch
- DistilBERT model (smaller and faster than BERT)
- Batch size of 16
- Maximum sequence length of 128 tokens

### Adjusting Parameters

You can adjust various parameters to customize your runs:

#### Model Selection

```bash
# Use a different pre-trained model
python main.py --model_name bert-base-uncased --num_epochs 3

# Try a different model architecture
python main.py --model_type adversarial --model_name roberta-base
```

Available model types:
- `baseline`: Standard transformer classifier
- `adversarial`: Adversarial training with media source information
- `triplet`: Triplet loss-based model

#### Training Parameters

```bash
# Increase epochs for better performance
python main.py --num_epochs 5

# Adjust batch size based on your GPU memory
python main.py --batch_size 8  # For smaller GPU
python main.py --batch_size 32  # For larger GPU

# Change learning rate
python main.py --learning_rate 2e-5
```

#### Data Processing

```bash
# Use media-based split instead of random split
python main.py --split_type media

# Adjust maximum sequence length
python main.py --max_length 256  # Longer sequences (more context, slower training)
python main.py --max_length 64   # Shorter sequences (less context, faster training)
```

## 3. Using Third-Party Language Models

You can use any model from Hugging Face's model hub by specifying the model name:

```bash
# Examples:
python main.py --model_name microsoft/deberta-base
python main.py --model_name google/electra-small-discriminator
python main.py --model_name facebook/bart-base
```

For larger models, you may need to reduce batch size:

```bash
python main.py --model_name google/flan-t5-base --batch_size 4 --gradient_accumulation_steps 4
```

## 4. Full Example for Comprehensive Training

After confirming everything works with the quick test, you can run a more comprehensive training:

```bash
python main.py \
  --model_name roberta-base \
  --model_type baseline \
  --num_epochs 3 \
  --batch_size 16 \
  --max_length 256 \
  --learning_rate 3e-5 \
  --split_type random \
  --output_dir ./results/roberta-baseline
```

## 5. Monitoring Training

The training progress will be displayed in the console. After training completes, results will be saved to the specified output directory.

## 6. Troubleshooting

- **Out of memory errors**: Reduce batch size or max_length
- **Slow training**: Use a smaller model like distilbert-base-uncased
- **Package conflicts**: Make sure you're using a clean virtual environment

## 7. Available Models

Some recommended models to try:

- **Smaller/Faster**: 
  - distilbert-base-uncased
  - google/electra-small-discriminator
  - microsoft/deberta-v3-small

- **Medium Size**:
  - bert-base-uncased
  - roberta-base
  - microsoft/deberta-base

- **Larger/More Accurate**:
  - roberta-large
  - microsoft/deberta-v3-large
  - facebook/bart-large