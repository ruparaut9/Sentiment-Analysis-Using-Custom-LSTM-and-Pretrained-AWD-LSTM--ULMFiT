# ================================
# Final Evaluation & Comparison Script
# ================================
# Evaluates Custom LSTM and ULMFiT models on IMDb test set,
# compares metrics side by side, and saves results + plots.
# ================================

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fastai imports for ULMFiT
from fastai.text.all import *

# Import custom dataset + model for LSTM
from utils.preprocessing import build_vocab, IMDBCsvDataset
from models.custom_lstm import SentimentLSTM

# --- Cleaning function (for ULMFiT) ---
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe:
    - Keeps only 'text' and 'label' columns
    - Strips whitespace
    - Converts labels to numeric (0/1)
    - Drops invalid rows
    """
    df = df[['text','label']].copy()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0]
    if df['label'].dtype == 'O':
        mapping = {'neg':0,'pos':1,'negative':0,'positive':1}
        df['label'] = df['label'].str.lower().map(mapping)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0,1])].reset_index(drop=True)
    return df

def evaluate_results():
    path = Path("D:/Projects/NLP_Sentiment_Analysis/data")

    # ================================
    # Evaluate ULMFiT
    # ================================
    train_df = clean_df(pd.read_csv(path/'imdb_train.csv'))
    test_df = clean_df(pd.read_csv(path/'imdb_test.csv'))

    dls = TextDataLoaders.from_df(train_df, text_col='text', label_col='label',
                                  valid_pct=0.2, seed=42, bs=64, num_workers=0)

    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5)
    learn.load("D:/Projects/NLP_Sentiment_Analysis/models/awd_lstm_ulmfit_imdb")

    # Build test dataloader with labels
    test_dl = dls.test_dl(test_df, with_labels=True)
    preds, targets = learn.get_preds(dl=test_dl)

    # Convert predictions to class labels
    y_pred = preds.argmax(dim=1)
    y_true = targets

    # Compute metrics using scikit-learn
    ulmfit_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "training_time": "approx 5 hours"   # updated
    }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (ULMFiT)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0,1], ["Negative","Positive"])
    plt.yticks([0,1], ["Negative","Positive"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="red")
    plt.savefig("D:/Projects/NLP_Sentiment_Analysis/models/confusion_matrix_ulmfit.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, preds[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (ULMFiT)")
    plt.legend(loc="lower right")
    plt.savefig("D:/Projects/NLP_Sentiment_Analysis/models/roc_curve_ulmfit.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, preds[:,1])
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (ULMFiT)")
    plt.legend(loc="lower left")
    plt.savefig("D:/Projects/NLP_Sentiment_Analysis/models/pr_curve_ulmfit.png")
    plt.close()

    # ================================
    # Evaluate Custom LSTM
    # ================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab("D:/Projects/NLP_Sentiment_Analysis/data/imdb_train.csv")
    test_dataset = IMDBCsvDataset("D:/Projects/NLP_Sentiment_Analysis/data/imdb_test.csv", vocab)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = SentimentLSTM(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load("D:/Projects/NLP_Sentiment_Analysis/best_model.pth"))
    model.eval()

    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            preds_batch = (outputs >= 0.5).long()
            test_correct += (preds_batch == labels.long()).sum().item()
            test_total += labels.size(0)
            all_preds.extend(preds_batch.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    custom_acc = test_correct / test_total
    custom_prec = precision_score(all_labels, all_preds)
    custom_rec = recall_score(all_labels, all_preds)
    custom_f1 = f1_score(all_labels, all_preds)

    custom_lstm_metrics = {
        "accuracy": custom_acc,
        "precision": custom_prec,
        "recall": custom_rec,
        "f1": custom_f1,
        "training_time": "approx 2 hours"   # updated
    }

    # ================================
    # Compare Results
    # ================================
    print("\nComparison of Models:")
    print(f"{'Metric':<12}{'Custom LSTM':<15}{'ULMFiT':<15}")
    print("-"*42)
    for metric in ["accuracy","precision","recall","f1"]:
        print(f"{metric:<12}{custom_lstm_metrics[metric]:<15.4f}{ulmfit_metrics[metric]:<15.4f}")
    print(f"{'Training Time':<12}{custom_lstm_metrics['training_time']:<15}{ulmfit_metrics['training_time']:<15}")

    # Bar chart comparison
    labels = ["Accuracy","Precision","Recall","F1 Score"]
    custom_values = [custom_lstm_metrics[m] for m in ["accuracy","precision","recall","f1"]]
    ulmfit_values = [ulmfit_metrics[m] for m in ["accuracy","precision","recall","f1"]]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(8,6))
    plt.bar([i-width/2 for i in x], custom_values, width, label="Custom LSTM")
    plt.bar([i+width/2 for i in x], ulmfit_values, width, label="ULMFiT")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Model Comparison: Custom LSTM vs ULMFiT")
    plt.legend()
    plt.ylim(0,1)
    plt.savefig("D:/Projects/NLP_Sentiment_Analysis/models/model_comparison.png")
    plt.close()

    # ================================
    # Save metrics to text file
    # ================================
    with open("D:/Projects/NLP_Sentiment_Analysis/models/evaluation_summary.txt","w") as f:
        f.write("ULMFiT Metrics:\n")
        for k,v in ulmfit_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nCustom LSTM Metrics:\n")
        for k,v in custom_lstm_metrics.items():
            f.write(f"{k}: {v}\n")

    print("\nFindings saved to evaluation_summary.txt and plots saved as PNGs.")

if __name__ == "__main__":
    evaluate_results()