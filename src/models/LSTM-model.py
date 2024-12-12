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

from src.preprocessing.preprocess_dataframe import preprocess_dataframe
from src.preprocessing.text_cleaning import clean_text
from src.preprocessing.vocabulary_builder import VocabularyBuilder

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
train_csv_path = "/Dataset/SemEval16/Train/Restaurants_Train.csv"
test_csv_path = "/Dataset/SemEval16/Test/Restaurants_Test.csv"

restaurant_df_train = pd.read_csv(train_csv_path, encoding='utf8')
test_df = pd.read_csv(test_csv_path, encoding='utf8')

# Combine Train and Test Data
df = pd.concat([restaurant_df_train, test_df], ignore_index=True)
logger.info(f"Combined dataset shape: {df.shape}")


# Apply Preprocessing
new_df = preprocess_dataframe(df)
new_df.head()



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

# Aspect LSTM Model
class AspectLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.5):
        super(AspectLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.sentence_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.aspect_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        concat_dim = hidden_dim * 4  # 2 (bidirectional) * 2 (sentence and aspect)
        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, sentence: torch.Tensor, aspect: torch.Tensor) -> torch.Tensor:
        sentence_embedded = self.embedding(sentence)
        aspect_embedded = self.embedding(aspect)
        sentence_output, _ = self.sentence_lstm(sentence_embedded)
        aspect_output, _ = self.aspect_lstm(aspect_embedded)
        # Concatenate the final forward and backward hidden states
        sentence_final = torch.cat((sentence_output[:, -1, :self.sentence_lstm.hidden_size],
                                    sentence_output[:, 0, self.sentence_lstm.hidden_size:]), dim=1)
        aspect_final = torch.cat((aspect_output[:, -1, :self.aspect_lstm.hidden_size],
                                  aspect_output[:, 0, self.aspect_lstm.hidden_size:]), dim=1)
        combined = torch.cat((sentence_final, aspect_final), dim=1)
        output = self.dropout(combined)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return output

# Training Function
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int, device: torch.device) -> Dict:
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0
        for batch_idx, (texts, aspects, labels) in enumerate(train_loader):
            texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, aspects)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        logger.info(f"Epoch {epoch}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for texts, aspects, labels in val_loader:
                texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)
                outputs = model(texts, aspects)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        logger.info(f"Epoch {epoch}/{num_epochs} - Validation Loss: {avg_val_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%")

        # Check for Improvement
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"Validation accuracy improved to {best_val_acc:.2f}%. Model saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation accuracy. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    return history

# Evaluation Function
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for texts, aspects, labels in test_loader:
            texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)
            outputs = model(texts, aspects)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_predictions, digits=4, target_names=['neutral', 'positive', 'negative', 'conflict'])
    logger.info("\nClassification Report:\n" + report)

    return {'predictions': all_predictions, 'true_labels': all_labels, 'classification_report': report}

# Plotting Functions
def plot_loss(history: Dict) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')  # Save the plot as a PNG file
    plt.show()

def plot_accuracy(history: Dict) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='green', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')  # Save the plot as a PNG file
    plt.show()

def plot_confusion_matrix(true_labels: List[int], predictions: List[int], classes: List[str]) -> None:
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the plot as a PNG file
    plt.show()

def plot_class_distribution(labels: List[int], classes: List[str]) -> None:
    label_counts = Counter(labels)
    counts = [label_counts[i] for i in range(len(classes))]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=classes, y=counts, palette='viridis')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    for index, value in enumerate(counts):
        plt.text(index, value, str(value), ha='center', va='bottom')
    plt.savefig('class_distribution.png')  # Save the plot as a PNG file
    plt.show()

# Initialization of Variables
torch.manual_seed(42)
np.random.seed(42)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

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

# Initialize the Model
model = AspectLSTM(
    vocab_size=len(vocab_builder.word2idx),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=4,  # neutral, positive, negative, conflict
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)
logger.info("Model initialized.")

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the Model
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,  # Using test set as validation for simplicity
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=device
)

# Plot Training and Validation Loss
plot_loss(history)

# Plot Validation Accuracy
plot_accuracy(history)

# Evaluate the Model
evaluation_results = evaluate_model(model, test_loader, device)

# Plot Confusion Matrix
classes = ['neutral', 'positive', 'negative', 'conflict']
plot_confusion_matrix(evaluation_results['true_labels'], evaluation_results['predictions'], classes)

# Optional: Plot Class Distribution
plot_class_distribution(y, classes)

# End of Script
