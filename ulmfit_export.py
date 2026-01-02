# ulmfit_export.py

from fastai.text.all import *
import pandas as pd
from pathlib import Path

def export_ulmfit():
    path = Path("D:/Projects/NLP_Sentiment_Analysis/data")
    train_df = pd.read_csv(path/'imdb_train.csv')

    # Disable multiprocessing (num_workers=0) to avoid Windows spawn issues
    dls = TextDataLoaders.from_df(
        train_df,
        text_col='text',
        label_col='label',
        valid_pct=0.2,
        seed=42,
        bs=64,
        num_workers=0
    )

    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        drop_mult=0.5,
        metrics=[accuracy, Precision(), Recall(), F1Score()]
    )

    learn.load("awd_lstm_ulmfit_imdb")
    learn.export("awd_lstm_ulmfit_imdb.pkl")
    print("Exported ULMFiT learner as awd_lstm_ulmfit_imdb.pkl")

if __name__ == "__main__":
    export_ulmfit()
