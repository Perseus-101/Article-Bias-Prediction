# Article Bias Prediction

This repository is a fork of [ramybaly/Article-Bias-Prediction](https://github.com/ramybaly/Article-Bias-Prediction). My goal is to enhance the original work by implementing and evaluating Triplet Loss Pre-training (TLP) for political bias detection. This enhancement aims to improve the model's ability to capture nuanced political biases in news articles through better representation learning using pre-trained models from HuggingFace.

## Dataset
The articles crawled from www.allsides.com are available in the `./data` folder, along with the different evaluation splits.

The dataset consists of a total of 37,554 articles. Each article is stored as a `JSON` object in the `./data/jsons` directory, and contains the following fields:
1. **ID**: an alphanumeric identifier.
2. **topic**: the topic being discussed in the article.
3. **source**: the name of the articles's source *(example: New York Times)*
4. **source_url**: the URL to the source's homepage *(example: www.nytimes.com)*
5. **url**: the link to the actual article.
6. **date**: the publication date of the article.
7. **authors**: a comma-separated list of the article's authors.
8. **title**: the article's title.
9. **content_original**: the original body of the article, as returned by the `newspaper3k` Python library.
10. **content**: the processed and tokenized content, which is used as input to the different models.
11. **bias_text**: the label of the political bias annotation of the article (left, center, or right).
12. **bias**: the numeric encoding of the political bias of the article (0, 1, or 2).

The `./data/splits` directory contains two types of splits: **random** and **media-based**.

## Setup

### Virtual Environment Setup

You can choose either Conda or venv:

#### Using Conda
```bash
# Create and activate environment
conda create -n bias-detection python=3.9
conda activate bias-detection

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

#### Using venv
```bash
# Create and activate environment
python -m venv bias-env
bias-env\Scripts\activate  # Windows
source bias-env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Models and Usage

### 1. Baseline Model
A standard transformer-based classifier using DistilBERT.

```bash
python main.py \
  --model_name distilbert-base-uncased \
  --model_type baseline \
  --num_epochs 3 \
  --batch_size 32 \
  --max_length 256 \
  --learning_rate 2e-5 \
  --split_type random
```
## Parameter Guidelines

Here are the recommended ranges and maximum values for different parameters across all model configurations:

### Common Parameters
- **model_name**: Any HuggingFace transformer model (recommended: distilbert-base-uncased)
- **max_length**: Maximum sequence length (recommended: 256-512)
- **split_type**: Either "random" or "media-based"

### Training Parameters
- **batch_size**: 8-32 (GPU memory dependent)
- **learning_rate**: 1e-5 to 5e-5
- **num_epochs**: 3-5 (can be increased for better performance)

### Pre-training Specific Parameters
- **pretrain_batch_size**: 8-32 (GPU memory dependent)
- **pretrain_epochs**: 3-5
- **pretrain_lr**: 1e-5 to 5e-5
- **finetune_batch_size**: 16-32
- **finetune_epochs**: 3-5
- **finetune_lr**: 1e-5 to 5e-5

### TLP-Specific Parameters
- **margin**: Margin for triplet loss (recommended: 0.5-1.0)
- **mining_strategy**: Strategy for mining triplets (options: "random", "hard", "semi-hard")

### 2. Triplet Loss Pre-training (TLP)
Enhances bias detection by learning article similarities using triplet loss. This approach uses HuggingFace transformer models as the backbone and fine-tunes them with triplet loss to better capture political bias in news articles.

#### Pre-training Phase
In this phase, the model learns to map articles with similar political biases closer together in the embedding space while pushing articles with different biases further apart.

```bash
python run_triplet_pretraining.py \
  --model_name distilbert-base-uncased \
  --pretrain_batch_size 8 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --max_length 256 \
  --split_type random \
  --margin 0.5 \
  --mining_strategy hard
```

#### Fine-tuning Phase
After pre-training, the model is fine-tuned on the classification task to predict the political bias of articles.

```bash
python run_triplet_pretraining.py \
  --model_name distilbert-base-uncased \
  --pretrain_batch_size 8 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --finetune_batch_size 16 \
  --finetune_epochs 5 \
  --finetune_lr 5e-5 \
  --max_length 256 \
  --split_type random
```

#### Using Other HuggingFace Models
You can experiment with different pre-trained models from HuggingFace by changing the `--model_name` parameter:

```bash
# Using BERT base
python run_triplet_pretraining.py \
  --model_name bert-base-uncased \
  --pretrain_batch_size 8 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --finetune_batch_size 16 \
  --finetune_epochs 5 \
  --finetune_lr 5e-5 \
  --max_length 256 \
  --split_type random

# Using RoBERTa base
python run_triplet_pretraining.py \
  --model_name roberta-base \
  --pretrain_batch_size 8 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --finetune_batch_size 16 \
  --finetune_epochs 5 \
  --finetune_lr 5e-5 \
  --max_length 256 \
  --split_type random
```

## Performance Comparison

### Results on Random Split (Ranked by Macro F1)

| Model | Approach | Macro F1 | Accuracy | MAE |
|-------|----------|----------|----------|-----|
| FacebookAI/roberta-base-baseline-max-performance | Baseline | 86.09 | 85.85 | 0.23 |
| FacebookAI/roberta-base-baseline | Baseline | 85.24 | 85.15 | 0.23 |
| roberta-base-baseline | Baseline | 85.02 | 84.92 | 0.23 |
| FacebookAI/roberta-base-baseline-high-learing-rate-5e-6 | Baseline | 81.05 | 80.85 | 0.31 |
| roberta-base-baseline | Baseline | 77.67 | 77.31 | 0.36 |
| FacebookAI/roberta-base-baseline | Baseline | 77.83 | 77.54 | 0.35 |
| roberta-base-triplet-pretrained | TLP | 78.7 | 77.77 | 0.38 |
| FacebookAI/roberta-base-triplet-pretrained | TLP | 76.19 | 75.69 | 0.4 |
| BERT Large | TLP | 76.43 | 75.85 | 0.40 |
| BERT Large | Baseline | 75.30 | 74.69 | 0.41 |
| PoliticalBiasBERT | Baseline | 75.08 | 75.00 | 0.40 |
| DistilBERT | Unsupervised SimCSE | 74.53 | 73.62 | 0.44 |
| BERT Base | Baseline | 73.25 | 72.92 | 0.42 |
| DistilBERT | TLP | 73.13 | 73.00 | 0.45 |
| DistilBERT | Hybrid Pretrained | 71.70 | 71.38 | 0.45 |
| DistilBERT | Baseline | 70.03 | 69.15 | 0.51 |

### Results on Media-Based Split (Ranked by Macro F1)

| Model | Approach | Macro F1 | Accuracy | MAE |
|-------|----------|----------|----------|-----|
| BERT Large | Baseline | 38.25 | 43.85 | 0.78 |
| BERT Large | TLP | 21.03 | 46.08 | 0.85 |

## Citation

```
@inproceedings{baly2020we,
  author      = {Baly, Ramy and Da San Martino, Giovanni and Glass, James and Nakov, Preslav},
  title       = {We Can Detect Your Bias: Predicting the Political Ideology of News Articles},
  booktitle   = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  series      = {EMNLP~'20},
  NOmonth     = {November},
  year        = {2020}
  pages       = {4982--4991},
  NOpublisher = {Association for Computational Linguistics}
}
```
