# ================================
# Project: Sentiment Analysis with BERT
# ================================
# Fine-tune BERT (bert-base-uncased) on IMDb dataset
# Includes train, validation, and test sets
# ================================

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress HuggingFace warnings about classifier head
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
def load_data(data_path):
    train_df = pd.read_csv(data_path / "imdb_train.csv")
    valid_df = pd.read_csv(data_path / "imdb_valid.csv")
    test_df = pd.read_csv(data_path / "imdb_test.csv")
    return train_df, valid_df, test_df

# -------------------------------
# Step 2: Tokenize Dataset
# -------------------------------
def tokenize_data(tokenizer, df, max_len=256):
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    labels = torch.tensor(df["label"].tolist())
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    return dataset

# -------------------------------
# Step 3: Train Function
# -------------------------------
def train(model, train_loader, valid_loader, optimizer, device, epochs=2):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # Validation after each epoch
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    return epoch_losses

# -------------------------------
# Step 4: Evaluation Function
# -------------------------------
def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1

# -------------------------------
# Step 5: Save Model & Results
# -------------------------------
def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def save_results(metrics, report_path):
    with open(report_path, "a") as f:
        f.write("\nBERT Metrics (Final Test Set):\n")
        f.write(f"Accuracy: {metrics[0]:.4f}\n")
        f.write(f"Precision: {metrics[1]:.4f}\n")
        f.write(f"Recall: {metrics[2]:.4f}\n")
        f.write(f"F1 Score: {metrics[3]:.4f}\n")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    data_path = Path("D:/Projects/NLP_Sentiment_Analysis/data")
    report_path = Path("D:/Projects/NLP_Sentiment_Analysis/reports/evaluation_summary.txt")
    save_path = Path("D:/Projects/NLP_Sentiment_Analysis/models/bert_model")

    # Load data
    train_df, valid_df, test_df = load_data(data_path)

    # Tokenizer & model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Tokenize
    train_dataset = tokenize_data(tokenizer, train_df)
    valid_dataset = tokenize_data(tokenizer, valid_df)
    test_dataset = tokenize_data(tokenizer, test_df)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    print("Train batches:", len(train_loader))
    print("Valid batches:", len(valid_loader))
    print("Test batches:", len(test_loader))

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train for 2 epochs with validation monitoring
    losses = train(model, train_loader, valid_loader, optimizer, device, epochs=2)

    # Final evaluation on test set
    metrics = evaluate(model, test_loader, device)
    print("Final Test Results:")
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1 Score: {metrics[3]:.4f}")

    # Save model & results
    save_model(model, tokenizer, save_path)
    save_results(metrics, report_path)

if __name__ == "__main__":
    main()
