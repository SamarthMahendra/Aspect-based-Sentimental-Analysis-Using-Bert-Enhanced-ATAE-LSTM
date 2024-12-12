import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
from collections import Counter
import torch.optim as optim
from typing import List, Tuple, Dict
import torch.nn.functional as F


# Import Necessary Libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import torch.optim as optim
from typing import List, Tuple, Dict
import logging

import pandas as pd
import ast
import re
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import nltk

import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK Tokenizer Resources
nltk.download('punkt')

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polarity Encoding Mapping (including 'conflict')
polarity_encoding = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
    'conflict': 3,  # Add 'conflict' to the mapping
}

# Load the Data
train_csv_path = "/Users/samarthmahendra/bioinfo/NLPprojectv2/Dataset/SemEval16/Train/Restaurants_Train.csv"
test_csv_path = "/Users/samarthmahendra/bioinfo/NLPprojectv2/Dataset/SemEval16/Test/Restaurants_Test.csv"

restaurant_df_train = pd.read_csv(train_csv_path, encoding='utf8')
test_df = pd.read_csv(test_csv_path, encoding='utf8')

# Combine Train and Test Data
df = pd.concat([restaurant_df_train, test_df], ignore_index=True)
logger.info(f"Combined dataset shape: {df.shape}")

# Function to Preprocess the DataFrame
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    processed_rows = []
    for _, row in df.iterrows():
        raw_text = row['raw_text']
        # Use ast.literal_eval instead of eval for safety
        try:
            aspect_terms = ast.literal_eval(row['aspectTerms'])
        except (ValueError, SyntaxError):
            aspect_terms = []
        for aspect in aspect_terms:
            polarity = aspect.get('polarity', 'none')
            if polarity != 'none':
                processed_rows.append({
                    'raw_text': raw_text,
                    'aspect_term': aspect['term'],
                    'polarity_encoded': polarity_encoding.get(polarity, 0)  # Default to 'neutral' if not found
                })
    processed_df = pd.DataFrame(processed_rows)
    logger.info(f"Processed dataframe shape: {processed_df.shape}")
    return processed_df

# Apply Preprocessing
new_df = preprocess_dataframe(df)
new_df.head()

# Function to Clean Text
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Apply Text Cleaning
new_df['raw_text'] = new_df['raw_text'].apply(clean_text)
new_df['aspect_term'] = new_df['aspect_term'].apply(clean_text)

# Prepare Features and Labels
X_raw_text = new_df['raw_text'].tolist()
X_aspect = new_df['aspect_term'].tolist()
y = new_df['polarity_encoded'].values

# Split the Data into Training and Testing Sets
X_train_text, X_test_text, X_train_aspect, X_test_aspect, y_train, y_test = train_test_split(
    X_raw_text, X_aspect, y, test_size=0.2, random_state=42, stratify=y
)
logger.info(f"Training set size: {len(X_train_text)}")
logger.info(f"Testing set size: {len(X_test_text)}")

# Vocabulary Builder Class
class VocabularyBuilder:
    def __init__(self, min_freq: int = 2):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts"""
        for text in texts:
            words = text.split()
            self.word_freq.update(words)

        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        logger.info(f"Vocabulary size: {len(self.word2idx)}")

    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to list of indices"""
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in text.split()]

# Dataset Class
class AspectSentimentDataset(Dataset):
    def __init__(self, texts: List[str], aspects: List[str], labels: np.ndarray, vocab: VocabularyBuilder):
        self.texts = texts
        self.aspects = aspects
        self.labels = torch.LongTensor(labels)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_indices = torch.LongTensor(self.vocab.text_to_indices(self.texts[idx]))
        aspect_indices = torch.LongTensor(self.vocab.text_to_indices(self.aspects[idx]))
        return text_indices, aspect_indices, self.labels[idx]

# Collate Function for DataLoader
def collate_fn(batch: List[Tuple]) -> Tuple:
    """Custom collate function to handle variable length sequences"""
    texts, aspects, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    aspects_padded = pad_sequence(aspects, batch_first=True, padding_value=0)
    return texts_padded, aspects_padded, torch.stack(labels)



class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        # Adjusting dimensions: hidden_dim is the size of BiLSTM output (hidden_dim * 2)
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, encoder_outputs: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim * 2)

        # Calculate attention scores
        attention_scores = self.attention(encoder_outputs).squeeze(-1)  # (batch_size, seq_len)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Apply attention weights to encoder outputs
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return weighted_output, attention_weights


class AspectAttentionLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2,
                 dropout: float = 0.5):
        super(AspectAttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Bidirectional LSTM layers
        self.sentence_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.aspect_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention layers for sentence and aspect
        self.sentence_attention = AttentionLayer(hidden_dim)
        self.aspect_attention = AttentionLayer(hidden_dim)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        # Using concatenated attended outputs from both LSTMs
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def create_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create mask for padding tokens (0)"""
        return (tensor != 0).float()

    def forward(self, sentence: torch.Tensor, aspect: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Create masks for padding
        sentence_mask = self.create_mask(sentence)
        aspect_mask = self.create_mask(aspect)

        # Embed inputs
        sentence_embedded = self.embedding(sentence)  # (batch_size, seq_len, embed_dim)
        aspect_embedded = self.embedding(aspect)  # (batch_size, aspect_len, embed_dim)

        # Process through BiLSTM
        sentence_outputs, _ = self.sentence_lstm(sentence_embedded)  # (batch_size, seq_len, hidden_dim*2)
        aspect_outputs, _ = self.aspect_lstm(aspect_embedded)  # (batch_size, aspect_len, hidden_dim*2)

        # Apply attention
        sentence_context, sentence_weights = self.sentence_attention(sentence_outputs, sentence_mask)
        aspect_context, aspect_weights = self.aspect_attention(aspect_outputs, aspect_mask)

        # Concatenate the attended outputs
        combined = torch.cat((sentence_context, aspect_context), dim=1)

        # Final classification
        output = self.dropout(combined)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)

        return output, sentence_weights, aspect_weights


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int,
                device: torch.device) -> Dict:
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (texts, aspects, labels) in enumerate(train_loader):
            texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(texts, aspects)  # Ignore attention weights during training
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for texts, aspects, labels in val_loader:
                texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)
                outputs, _, _ = model(texts, aspects)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_attention.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_predictions = []
    all_labels = []
    all_sentence_attention = []
    all_aspect_attention = []

    with torch.no_grad():
        for texts, aspects, labels in test_loader:
            texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)
            outputs, sentence_attention, aspect_attention = model(texts, aspects)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sentence_attention.extend(sentence_attention.cpu().numpy())
            all_aspect_attention.extend(aspect_attention.cpu().numpy())

    report = classification_report(all_labels, all_predictions, digits=4)
    print("\nClassification Report:")
    print(report)

    return {
        'predictions': all_predictions,
        'true_labels': all_labels,
        'sentence_attention': all_sentence_attention,
        'aspect_attention': all_aspect_attention,
        'classification_report': report
    }

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyperparameters
EMBED_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MIN_WORD_FREQ = 2


# Build Vocabulary
vocab_builder = VocabularyBuilder(min_freq=MIN_WORD_FREQ)
vocab_builder.build_vocab(X_train_text + X_train_aspect)

# Create Datasets
train_dataset = AspectSentimentDataset(
    texts=X_train_text,
    aspects=X_train_aspect,
    labels=y_train,
    vocab=vocab_builder
)
test_dataset = AspectSentimentDataset(
    texts=X_test_text,
    aspects=X_test_aspect,
    labels=y_test,
    vocab=vocab_builder
)

# Create Data Loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# Initialize model with attention
model = AspectAttentionLSTM(
    vocab_size=len(vocab_builder.word2idx),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=4,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=device
)

# Evaluate the model
evaluation_results = evaluate_model(model, test_loader, device)