# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from typing import Tuple, Dict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f'Using device: {device}')

# Define hyperparameters
EMBED_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 32
MAX_LEN = 100
NUM_CLASSES = 4  # neutral, positive, negative, conflict

# Define polarity encoding
polarity_encoding = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
    'conflict': 3,
}

# Define file paths
train_path = "/Users/samarthmahendra/bioinfo/NLPprojectv2/Dataset/SemEval16/Train/Restaurants_Train.csv"
test_path = "/Users/samarthmahendra/bioinfo/NLPprojectv2/Dataset/SemEval16/Test/Restaurants_Test.csv"

# Load the training and testing data
restaurant_df_train = pd.read_csv(train_path, encoding='utf8')
test_df = pd.read_csv(test_path, encoding='utf8')

# Combine train and test data for preprocessing
df = pd.concat([restaurant_df_train, test_df], ignore_index=True)


def preprocess_dataframe(df):
    """
    Preprocess the dataframe by extracting relevant information and encoding polarity.

    Args:
        df (pd.DataFrame): Combined train and test dataframe.

    Returns:
        pd.DataFrame: Processed dataframe with 'raw_text', 'aspect_term', and 'polarity_encoded'.
    """
    processed_rows = []
    for _, row in df.iterrows():
        raw_text = row['raw_text']
        aspect_terms = ast.literal_eval(row['aspectTerms'])  # Convert string to list
        for aspect in aspect_terms:
            if aspect['polarity'] != 'none':  # Filter out aspects with 'none' polarity
                processed_rows.append({
                    'raw_text': raw_text,
                    'aspect_term': aspect['term'],
                    'polarity_encoded': polarity_encoding[aspect['polarity']]
                })
    return pd.DataFrame(processed_rows)


# Apply preprocessing
new_df = preprocess_dataframe(df)

# Split into train and test sets with stratification to maintain class distribution
train_df, test_df = train_test_split(
    new_df,
    test_size=0.2,
    random_state=SEED,
    stratify=new_df['polarity_encoded']
)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class AspectDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['raw_text']
        aspect = row['aspect_term']
        label = row['polarity_encoded']

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize the aspect
        aspect_encoding = self.tokenizer.encode_plus(
            aspect,
            add_special_tokens=True,
            max_length=10,  # assuming aspect terms are short
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Squeeze to remove extra dimensions
        input_ids = encoding['input_ids'].squeeze()
        aspect_ids = aspect_encoding['input_ids'].squeeze()

        return input_ids, aspect_ids, torch.tensor(label, dtype=torch.long)


# Create datasets
train_dataset = AspectDataset(train_df, tokenizer, MAX_LEN)
val_dataset = AspectDataset(test_df, tokenizer, MAX_LEN)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")


class ATAEAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(ATAEAttentionLayer, self).__init__()
        # Accommodate concatenated hidden states and aspect embeddings
        self.attention = nn.Linear(hidden_dim * 4, 1)  # hidden_dim * 2 (BiLSTM) * 2 (hidden + aspect)

    def forward(self,
                encoder_outputs: torch.Tensor,
                aspect_embedding: torch.Tensor,
                mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_outputs: (batch_size, seq_len, hidden_dim * 2)
        # aspect_embedding: (batch_size, hidden_dim * 2)

        batch_size, seq_len, hidden_dim = encoder_outputs.size()

        # Repeat aspect_embedding across the sequence length
        aspect_repeated = aspect_embedding.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim * 2)

        # Concatenate encoder outputs with aspect embeddings
        combined = torch.cat([encoder_outputs, aspect_repeated], dim=2)  # (batch_size, seq_len, hidden_dim * 4)

        # Calculate attention scores
        attention_scores = self.attention(combined).squeeze(-1)  # (batch_size, seq_len)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Apply attention weights to encoder outputs
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(
            1)  # (batch_size, hidden_dim * 2)

        return weighted_output, attention_weights


class ATAELSTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 pretrained_embeddings: torch.Tensor = None):
        super(ATAELSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Embedding layer for both words and aspects
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Main LSTM
        self.lstm = nn.LSTM(
            embed_dim + embed_dim,  # word embedding + aspect embedding
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Aspect LSTM
        self.aspect_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ATAE attention layer
        self.attention = ATAEAttentionLayer(hidden_dim)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def create_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create mask for padding tokens (0)"""
        return (tensor != 0).float()

    def forward(self, sentence: torch.Tensor, aspect: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create masks
        sentence_mask = self.create_mask(sentence)

        # Embed inputs
        sentence_embedded = self.embedding(sentence)  # (batch_size, seq_len, embed_dim)
        aspect_embedded = self.embedding(aspect)  # (batch_size, aspect_len, embed_dim)

        # Get aspect representation
        aspect_output, (aspect_hidden, _) = self.aspect_lstm(aspect_embedded)
        # Concatenate forward and backward hidden states
        aspect_repr = torch.cat([aspect_hidden[-2], aspect_hidden[-1]], dim=1)  # (batch_size, hidden_dim * 2)

        # Repeat aspect embedding for each word in the sentence
        batch_size, seq_len, _ = sentence_embedded.size()
        aspect_repeated = aspect_embedded.mean(1).unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate word embeddings with repeated aspect embeddings
        combined_input = torch.cat([sentence_embedded, aspect_repeated], dim=2)

        # Process through LSTM
        lstm_outputs, _ = self.lstm(combined_input)

        # Apply attention
        attended_output, attention_weights = self.attention(
            lstm_outputs,
            aspect_repr,
            sentence_mask
        )

        # Final classification
        output = self.dropout(attended_output)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)

        return output, attention_weights


def train_atae_model(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     criterion: nn.Module,
                     optimizer: optim.Optimizer,
                     num_epochs: int,
                     device: torch.device) -> Dict:
    """
    Train the ATAE-LSTM model.

    Args:
        model (nn.Module): The ATAE-LSTM model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to train on.

    Returns:
        Dict: Training history containing losses and accuracies.
    """
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
            outputs, _ = model(texts, aspects)
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
                outputs, _ = model(texts, aspects)
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

        # Early Stopping and Checkpointing
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_atae.pth')
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return history


def evaluate_atae_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """
    Evaluate the ATAE-LSTM model.

    Args:
        model (nn.Module): The trained ATAE-LSTM model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to evaluate on.

    Returns:
        Dict: Evaluation results including predictions, true labels, attention weights, and classification report.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_attention_weights = []

    with torch.no_grad():
        for texts, aspects, labels in test_loader:
            texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)
            outputs, attention_weights = model(texts, aspects)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())

    report = classification_report(all_labels, all_predictions, digits=4)
    print("\nClassification Report:")
    print(report)

    return {
        'predictions': all_predictions,
        'true_labels': all_labels,
        'attention_weights': all_attention_weights,
        'classification_report': report
    }


# Initialize the model
# Use tokenizer's vocab size
vocab_size = tokenizer.vocab_size
model = ATAELSTM(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pretrained_embeddings=None  # Replace with pretrained_embeddings if available
).to(device)

# Compute class weights to handle class imbalance
y_train = train_df['polarity_encoded'].tolist()
class_counts = Counter(y_train)
print("Class Counts:", class_counts)

total_samples = len(y_train)
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
weights = torch.tensor([class_weights.get(i, 1.0) for i in range(NUM_CLASSES)], dtype=torch.float32).to(device)
print("Class Weights:", weights)

# Define loss function with class weights
criterion = nn.CrossEntropyLoss(weight=weights)

# Define optimizer with weight decay for L2 regularization
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Train the model
history = train_atae_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,  # Use validation set
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=device
)

# Load the best model
model.load_state_dict(torch.load('best_model_atae.pth'))
print("Best model loaded for evaluation.")

# Evaluate the model
evaluation_results = evaluate_atae_model(model, val_loader, device)



