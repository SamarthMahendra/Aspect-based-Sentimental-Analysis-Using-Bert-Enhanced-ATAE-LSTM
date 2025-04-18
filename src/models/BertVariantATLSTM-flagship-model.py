# Import necessary libraries
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import os

# Import custom preprocessing functions
# Ensure that these modules are available in your project
from src.preprocessing.preprocess_dataframe import preprocess_dataframe
from src.preprocessing.vocabulary_builder import VocabularyBuilder

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# -----------------------------
# 1. Data Preparation
# -----------------------------

# Polarity encoding mapping (including 'conflict')
polarity_encoding = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
    'conflict': 3,  # Added 'conflict' to the mapping
}


# Function to load and preprocess data
def load_and_preprocess_data(train_path, test_path):
    # Load the training and testing data
    restaurant_df_train = pd.read_csv(
        train_path,
        encoding='utf8'
    )
    test_df = pd.read_csv(
        test_path,
        encoding='utf8'
    )

    # Combine both train and test data for preprocessing
    df = pd.concat([restaurant_df_train, test_df], ignore_index=True)


    return preprocess_dataframe(df)


# Paths to the dataset
train_csv_path = "data/SemEval16/Train/Restaurants_Train.csv"
test_csv_path = "data/SemEval16/Test/Restaurants_Test.csv"

# Load and preprocess data
df = load_and_preprocess_data(train_csv_path, test_csv_path)

# Split into train, validation, and test sets with stratification
train_val_df, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df['polarity_encoded']
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1,
    random_state=42,
    stratify=train_val_df['polarity_encoded']
)

# Reset index after split
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# -----------------------------
# 2. Device Configuration
# -----------------------------

# Check for available device (preferably GPU)
if torch.backends.mps.is_available():
    device = torch.device('mps')  # For Apple Silicon Macs
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# -----------------------------
# 3. Tokenizer and BERT Model Setup
# -----------------------------

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# -----------------------------
# 4. Dataset and DataLoader
# -----------------------------

class BertAspectDataset(Dataset):
    """
    Custom Dataset for BERT-based Aspect Sentiment Analysis.
    """

    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row['raw_text']
        aspect = row['aspect_term']
        label = row['polarity_encoded']

        # Use tokenizer's ability to handle two separate texts
        encoding = self.tokenizer(
            text=sentence,
            text_pair=aspect,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),  # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(),  # [max_length]
            'token_type_ids': encoding['token_type_ids'].squeeze(),  # [max_length]
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Create dataset instances
train_dataset = BertAspectDataset(train_df, tokenizer, max_length=128)
val_dataset = BertAspectDataset(val_df, tokenizer, max_length=128)
test_dataset = BertAspectDataset(test_df, tokenizer, max_length=128)

# Create DataLoader instances
batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -----------------------------
# 5. Model Definition
# -----------------------------

class BERT_AttentionLSTM_Enhanced(nn.Module):
    """
    Integrated BERT with LSTM and Enhanced Attention for Aspect-Based Sentiment Analysis.
    """

    def __init__(self, bert_model, hidden_dim, num_classes):
        super(BERT_AttentionLSTM_Enhanced, self).__init__()
        self.bert = bert_model
        self.sentence_lstm = nn.LSTM(
            bert_model.config.hidden_size,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.aspect_lstm = nn.LSTM(
            bert_model.config.hidden_size,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.attention = nn.Linear(hidden_dim * 4, 1)  # Enhanced attention
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Pass combined sentence and aspect through BERT
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = bert_outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        # Split the sequence into sentence and aspect parts based on token_type_ids
        # token_type_ids == 0: sentence, ==1: aspect
        sentence_mask = token_type_ids == 0  # [batch, seq_len]
        aspect_mask = token_type_ids == 1  # [batch, seq_len]

        # Extract sentence embeddings
        sentence_embs = last_hidden_state * sentence_mask.unsqueeze(-1)  # [batch, seq_len, hidden_size]
        # Extract aspect embeddings
        aspect_embs = last_hidden_state * aspect_mask.unsqueeze(-1)  # [batch, seq_len, hidden_size]

        # Pass through LSTMs
        sentence_lstm_out, _ = self.sentence_lstm(sentence_embs)  # [batch, seq_len, hidden_dim*2]
        aspect_lstm_out, _ = self.aspect_lstm(aspect_embs)  # [batch, seq_len, hidden_dim*2]

        # Compute mean of aspect embeddings
        aspect_mean = torch.mean(aspect_lstm_out, dim=1).unsqueeze(1)  # [batch, 1, hidden_dim*2]

        # Tile aspect_mean to match sentence length
        aspect_tiled = aspect_mean.repeat(1, sentence_lstm_out.size(1), 1)  # [batch, sentence_len, hidden_dim*2]

        # Combine sentence and aspect embeddings
        combined_embs = torch.cat([sentence_lstm_out, aspect_tiled], dim=-1)  # [batch, sentence_len, hidden_dim*4]

        # Apply enhanced attention
        attention_scores = self.attention(combined_embs).squeeze(-1)  # [batch, sentence_len]
        alpha = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch, sentence_len, 1]
        attended = combined_embs * alpha  # [batch, sentence_len, hidden_dim*4]
        context = torch.sum(attended, dim=1)  # [batch, hidden_dim*4]

        # Pass through fully connected layer
        out = self.dropout(context)
        logits = self.fc(out)  # [batch, num_classes]

        return logits, alpha


# Initialize the enhanced model
hidden_dim = 256
num_classes = len(polarity_encoding)  # neutral, positive, negative, conflict
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BERT_AttentionLSTM_Enhanced(bert_model, hidden_dim, num_classes).to(device)

# -----------------------------
# 6. Training Setup
# -----------------------------

# Calculate class weights to handle class imbalance
y_train = train_df['polarity_encoded'].tolist()
class_counts = Counter(y_train)
print("Class Counts:", class_counts)  # Debugging line

total_samples = len(y_train)
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
weights = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)], dtype=torch.float32).to(device)
print("Class Weights:", weights)  # Debugging line

# Define loss function with class weights
criterion = nn.CrossEntropyLoss(weight=weights)

# Define optimizer with a higher learning rate
optimizer = optim.AdamW(model.parameters(), lr=3e-5)

# Define scheduler
EPOCHS = 10
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# -----------------------------
# 7. Training and Evaluation Functions
# -----------------------------

from torch.nn.utils import clip_grad_norm_


def train_epoch(model, data_loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs, _ = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    epoch_loss = running_loss / total_train
    epoch_acc = correct_train / total_train

    return epoch_loss, epoch_acc


def eval_model(model, data_loader, criterion, device):
    """
    Evaluates the model on a validation or test set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs, _ = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(labels_list, preds_list, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1, preds_list, labels_list


# -----------------------------
# 8. Training Loop with Early Stopping
# -----------------------------

# Early Stopping parameters
best_f1 = 0
patience = 3
counter = 0
best_model_path = 'best_model.pt'

# Lists to store metrics
train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
val_f1_list = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Validate
    val_loss, val_acc, val_f1, _, _ = eval_model(model, val_loader, criterion, device)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    val_f1_list.append(val_f1)
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, Validation F1: {val_f1:.4f}")

    # Early Stopping Check
    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved.")
    else:
        counter += 1
        print(f"No improvement in F1 for {counter} epoch(s).")
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model
model.load_state_dict(torch.load(best_model_path))

# -----------------------------
# 9. Inference on Sample Sentence
# -----------------------------

test_sentence = (
    "The decor is night tho...but they REALLY need to clean that vent in the ceiling...its quite "
    "un-appetizing, and kills your effort to make this place look sleek and modern."
)
test_aspects = ["place", "decor", "vent"]


def preprocess_input_bert_embeddings(sentence, aspect, tokenizer, max_length=128):
    """
    Preprocess the input sentence and aspect to obtain BERT inputs.

    Args:
        sentence (str): The raw sentence.
        aspect (str): The aspect term.
        tokenizer: BERT tokenizer.
        max_length (int): Maximum token length.

    Returns:
        dict: Dictionary containing 'input_ids', 'attention_mask', and 'token_type_ids'.
    """
    encoding = tokenizer(
        text=sentence,
        text_pair=aspect,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].squeeze().to(device),
        'attention_mask': encoding['attention_mask'].squeeze().to(device),
        'token_type_ids': encoding['token_type_ids'].squeeze().to(device)
    }


# Invert polarity encoding for interpretation
inv_polarity = {v: k for k, v in polarity_encoding.items()}

# Set model to evaluation mode
model.eval()
print("\nInference on Sample Sentence:")
with torch.no_grad():
    for aspect in test_aspects:
        # Preprocess the input sentence and aspect
        inputs = preprocess_input_bert_embeddings(test_sentence, aspect, tokenizer, max_length=128)
        input_ids = inputs['input_ids'].unsqueeze(0)  # [1, max_length]
        attention_mask = inputs['attention_mask'].unsqueeze(0)  # [1, max_length]
        token_type_ids = inputs['token_type_ids'].unsqueeze(0)  # [1, max_length]

        # Predict using the trained model
        outputs, attention_weights = model(input_ids, attention_mask, token_type_ids)
        pred_label = torch.argmax(outputs, dim=1).item()

        print(f"Aspect: '{aspect}', Predicted Sentiment: {inv_polarity[pred_label]}")


# -----------------------------
# 10. Visualization & Metrics
# -----------------------------

# 1. Class Distribution Plot
def plot_class_distribution(train_df, val_df, test_df, polarity_encoding):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(x='polarity_encoded', data=train_df, ax=ax[0], palette='viridis')
    ax[0].set_title('Training Set Class Distribution')
    ax[0].set_xlabel('Polarity')
    ax[0].set_ylabel('Count')
    ax[0].set_xticklabels([k for k, v in sorted(polarity_encoding.items(), key=lambda item: item[1])])

    sns.countplot(x='polarity_encoded', data=val_df, ax=ax[1], palette='viridis')
    ax[1].set_title('Validation Set Class Distribution')
    ax[1].set_xlabel('Polarity')
    ax[1].set_ylabel('Count')
    ax[1].set_xticklabels([k for k, v in sorted(polarity_encoding.items(), key=lambda item: item[1])])

    sns.countplot(x='polarity_encoded', data=test_df, ax=ax[2], palette='viridis')
    ax[2].set_title('Testing Set Class Distribution')
    ax[2].set_xlabel('Polarity')
    ax[2].set_ylabel('Count')
    ax[2].set_xticklabels([k for k, v in sorted(polarity_encoding.items(), key=lambda item: item[1])])

    plt.tight_layout()
    plt.show()


plot_class_distribution(train_df, val_df, test_df, polarity_encoding)


# 2. Confusion Matrix & Classification Report on Test Set
def evaluate_on_test_set(model, test_loader, criterion, device, polarity_encoding):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs, _ = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(labels_list, preds_list)

    # Define labels in order
    labels_order = sorted(polarity_encoding, key=polarity_encoding.get)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["neutral", 'positive', "negative"],
                yticklabels=["neutral", 'positive', "negative"])
    plt.title('Confusion Matrix on Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Generate and print classification report
    print("\nClassification Report on Test Set:")
    print(classification_report(labels_list, preds_list, target_names=["neutral", 'positive', "negative"]))

    # Overall Accuracy
    overall_acc = accuracy_score(labels_list, preds_list)
    print(f"Overall Test Accuracy: {overall_acc:.4f}")


evaluate_on_test_set(model, test_loader, criterion, device, polarity_encoding)


# 3. Training and Validation Metrics Plot
def plot_training_history(train_loss, train_acc, val_loss, val_acc, val_f1):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy and F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.plot(epochs, val_f1, 'g-', label='Validation F1 Score')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(train_loss_list, train_acc_list, val_loss_list, val_acc_list, val_f1_list)

