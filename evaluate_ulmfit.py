from fastai.text.all import *
from pathlib import Path
import pandas as pd

# Same cleaning function
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
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

def evaluate_ulmfit():
    path = Path("D:/Projects/NLP_Sentiment_Analysis/data")

    # Recreate DataLoaders (same as training)
    train_df = clean_df(pd.read_csv(path/'imdb_train.csv'))
    dls = TextDataLoaders.from_df(
        train_df,
        text_col='text',
        label_col='label',
        valid_pct=0.2,
        seed=42,
        bs=64,
        num_workers=0
    )

    # Rebuild learner and load saved weights
    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        drop_mult=0.5,
        metrics=[accuracy, Precision(), Recall(), F1Score()]
    )
    learn.load("awd_lstm_ulmfit_imdb")   # load trained model

    # Load and clean test data
    test_df = clean_df(pd.read_csv(path/'imdb_test.csv'))

    #  Important: include labels
    test_dl = dls.test_dl(test_df, with_labels=True)

    # Evaluate
    test_results = learn.validate(dl=test_dl)
    print("\nTest results [loss, accuracy, precision, recall, f1]:", test_results)

if __name__ == "__main__":
    evaluate_ulmfit()
