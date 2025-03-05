# Article Bias Prediction

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

### 2. Triplet Loss Pre-training (TLP)
Enhances bias detection by learning article similarities using triplet loss.

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

### 3. SimCSE Pre-training
Leverages contrastive learning with SimCSE for better article representations.

```bash
python run_simcse_pretraining.py \
  --model_name distilbert-base-uncased \
  --pretrain_batch_size 32 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --finetune_batch_size 32 \
  --finetune_epochs 3 \
  --finetune_lr 5e-5 \
  --max_length 256 \
  --split_type random \
  --temperature 0.05
```

Add `--supervised` flag for supervised SimCSE.

### 4. Hybrid Approach (TLP + SimCSE)
Combines triplet loss and SimCSE for comprehensive article understanding.

```bash
python run_hybrid_pretraining.py \
  --model_name distilbert-base-uncased \
  --pretrain_batch_size 16 \
  --pretrain_epochs 3 \
  --pretrain_lr 2e-5 \
  --finetune_batch_size 32 \
  --finetune_epochs 3 \
  --finetune_lr 5e-5 \
  --max_length 256 \
  --split_type random \
  --triplet_weight 1.0 \
  --simcse_weight 1.0 \
  --ce_weight 0.1
```

## Performance Comparison

Results on random split using DistilBERT:

| Model | Macro F1 | Accuracy | MAE |
|-------|----------|----------|-----|
| Baseline | 70.03 | 69.15 | 0.51 |
| TLP | 73.13 | 73.00 | 0.45 |
| SimCSE | 74.53 | 73.62 | 0.44 |
| Hybrid | 71.70 | 71.38 | 0.45 |

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
