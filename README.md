# BERT with Attention and LSTM for Aspect-Based Sentiment Analysis

This project implements an Aspect-Based Sentiment Analysis (ABSA) model using a combination of BERT, LSTM, and an enhanced Attention mechanism. The model is designed to analyze restaurant reviews, identifying sentiments (neutral, positive, negative, conflict) associated with specific aspects of the review (e.g., decor, vent, place).

## Features

- **BERT Integration:** Utilizes the pre-trained BERT model for rich contextual embeddings
- **Bidirectional LSTM:** Captures sequential dependencies in both forward and backward directions
- **Enhanced Attention Mechanism:** Focuses on relevant parts of the sentence concerning the aspect term
- **Class Imbalance Handling:** Implements class weighting to address uneven class distributions
- **Early Stopping:** Prevents overfitting by halting training when performance plateaus
- **Comprehensive Evaluation:** Includes metrics like accuracy, F1-score, and confusion matrix
- **Visualization:** Provides plots for class distribution and model performance

## Dataset

The model is trained and evaluated on the [SemEval-2016 Task 5](https://competitions.codalab.org/competitions/17751) dataset, specifically the Restaurant Reviews subset.

### Data Files
- **Training Data:** `Restaurants_Train.csv`
- **Testing Data:** `Restaurants_Test.csv`

### Data Preprocessing
- Extracts relevant information such as `raw_text`, `aspect_term`, and encodes `polarity`
- Filters out aspects with 'none' polarity

## Installation

### Prerequisites
- Python 3.7 or higher
- `pip` package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ABSA-BERT-LSTM-Attention.git
cd ABSA-BERT-LSTM-Attention
```

2. Create and activate virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For manual installation of packages:
```bash
pip install torch transformers sklearn pandas numpy tqdm matplotlib seaborn
```

### CUDA Support (Optional)
Ensure CUDA is installed and compatible with your PyTorch version for GPU acceleration.

## Usage

### Data Preparation
Place the dataset CSV files in their respective directories:
```
/Dataset/SemEval16/Train/Restaurants_Train.csv
/Dataset/SemEval16/Test/Restaurants_Test.csv
```

### Running the Model
Execute the main script:
```bash
python models/BertVariantATLSTM-flagship-model.py
```

The script performs:
1. Data preprocessing
2. Device configuration (GPU/CPU)
3. Model training
4. Evaluation
5. Inference
6. Visualization

## Model Architecture

The model combines three main components:

1. **BERT Embeddings**
   - Processes input sentences and aspect terms
   - Generates contextual embeddings

2. **Bidirectional LSTM**
   - Captures sequential dependencies
   - Processes both sentence and aspect embeddings

3. **Attention Mechanism**
   - Computes relevance scores
   - Focuses on aspect-specific sentence parts

4. **Output Layer**
   - Fully connected layer for classification
   - Dropout for regularization

## Training

### Configuration
- Epochs: 10
- Batch Size: 16
- Learning Rate: 3e-5
- Optimizer: AdamW
- Scheduler: Linear schedule with warmup
- Loss Function: CrossEntropyLoss with class weights
- Early Stopping: Based on validation F1-score (patience=3)

## Evaluation

The model's performance is evaluated using:
- Accuracy
- F1-Score (Weighted)
- Confusion Matrix
- Classification Report

## Inference

Sample inference output:
```
Inference on Sample Sentence:
Aspect: 'place', Predicted Sentiment: neutral
Aspect: 'decor', Predicted Sentiment: negative
Aspect: 'vent', Predicted Sentiment: negative
```

## Project Structure

```
NLPprojectv2/
│
├── .ipynb_checkpoints/
├── .venv/
├── data/
├── Dataset/
│   ├── SemEval14/
│   ├── SemEval15/
│   └── SemEval16/
├── models/
│   ├── AT-LSTM-model.py
│   ├── ATAE_LSTM-model.py
│   ├── BertVariantATLSTM-flagship-model.py
│   └── LSTM-model.py
├── notebooks/
│   └── NLP_final_Proejct_Notebook.ipynb
├── best_model.pt
├── best_model.pth
├── best_model_atae.pth
├── best_model_attention.pth
├── best_model_bert_atae.pth
└── Dataset.zip
```


## Acknowledgments

- BERT by Google Research
- SemEval-2016 Task 5 for providing the dataset
- Contributors and the open-source community