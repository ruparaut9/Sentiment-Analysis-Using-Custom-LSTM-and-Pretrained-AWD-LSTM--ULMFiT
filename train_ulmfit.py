# ================================
# Sentiment Analysis with AWD-LSTM (ULMFiT)
# ================================
# This script fine-tunes a pretrained AWD-LSTM model (ULMFiT) on your IMDb dataset.
# It uses fastai's high-level API for text classification.
# Key steps:
#   1. Load and clean the dataset
#   2. Create DataLoaders
#   3. Build a learner with pretrained AWD-LSTM
#   4. Fine-tune the model
#   5. Evaluate on validation and test sets
#   6. Save the trained model

from fastai.text.all import *
from pathlib import Path
import pandas as pd

# --- STEP 1: Data Cleaning Function ---
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame:
    - Keeps only 'text' and 'label' columns
    - Removes empty text rows
    - Converts labels to integers {0,1}
    - Drops invalid rows
    """
    df = df[['text','label']].copy()

    # Clean text: strip whitespace, drop empty rows
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0]

    # Normalize labels
    if df['label'].dtype == 'O':  # if labels are strings
        mapping = {'neg':0,'pos':1,'negative':0,'positive':1}
        df['label'] = df['label'].str.lower().map(mapping)

    # Convert to numeric, drop invalids
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Keep only 0/1 labels
    df = df[df['label'].isin([0,1])].reset_index(drop=True)

    return df

# --- STEP 2: Training Function ---
def train_ulmfit():
    # Path to your dataset folder
    path = Path("D:/Projects/NLP_Sentiment_Analysis/data")

    # Load and clean training data
    train_df = clean_df(pd.read_csv(path/'imdb_train.csv'))
    print("Train label distribution:", train_df['label'].value_counts().to_dict())

    # Create DataLoaders (fastai handles tokenization, numericalization, batching)
    dls = TextDataLoaders.from_df(
        train_df,
        text_col='text',       # column with review text
        label_col='label',     # column with sentiment labels
        valid_pct=0.2,         # 20% of data used for validation
        seed=42,               # reproducibility
        bs=64,                 # batch size
        num_workers=0          # safer on Windows
    )

    # --- STEP 3: Build Learner ---
    # AWD-LSTM is a pretrained language model (ULMFiT)
    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        drop_mult=0.5,         # regularization multiplier
        metrics=[accuracy, Precision(), Recall(), F1Score()]
    )

    # --- STEP 4: Train the Model ---
    # fine_tune(4) = 1 epoch with frozen LM + 3 epochs unfrozen
    print("\nStarting training...\n")
    learn.fine_tune(4)

    # --- STEP 5: Evaluate ---
    print("\nValidation results:", learn.validate())

    # Load and clean test data
    test_df = clean_df(pd.read_csv(path/'imdb_test.csv'))
    test_dl = dls.test_dl(test_df)
    print("\nTest results:", learn.validate(dl=test_dl))

    # --- STEP 6: Save Model ---
    # This saves weights to models/awd_lstm_ulmfit_imdb.pth
    learn.save("awd_lstm_ulmfit_imdb")
    print("\nModel saved as 'awd_lstm_ulmfit_imdb.pth' in the models folder.")

# --- Entry Point ---
if __name__ == "__main__":
    train_ulmfit()
