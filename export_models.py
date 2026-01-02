import torch
import pickle
import logging
import os
import time
from fastai.text.all import *
from transformers import BertTokenizer, BertForSequenceClassification

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Import your custom LSTM class ---
import sys
sys.path.append(os.path.abspath("D:/Projects/NLP_Sentiment_Analysis/models"))
from custom_lstm import SentimentLSTM

def timed_step(step_name, func):
    """Utility to measure execution time of each step"""
    logging.info(f"Starting {step_name}...")
    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        logging.info(f" {step_name} completed in {elapsed:.2f} seconds")
    except Exception as e:
        elapsed = time.time() - start
        logging.error(f" {step_name} failed after {elapsed:.2f} seconds: {e}")

# -------------------------------
# 1. Export Custom LSTM
# -------------------------------
def export_custom_lstm():
    state_dict = torch.load(r"D:\Projects\NLP_Sentiment_Analysis\best_model.pth")
    vocab_size = state_dict["embedding.weight"].shape[0]
    custom_model = SentimentLSTM(vocab_size=vocab_size)
    custom_model.load_state_dict(state_dict)
    custom_model.eval()
    torch.save(custom_model, r"D:\Projects\NLP_Sentiment_Analysis\models\custom_lstm.pkl")

timed_step("Custom LSTM export", export_custom_lstm)

# -------------------------------
# 2. Export ULMFiT
# -------------------------------
def export_ulmfit():
    ulmfit_pkl = r"D:\Projects\NLP_Sentiment_Analysis\models\awd_lstm_ulmfit_imdb.pkl"
    if os.path.exists(ulmfit_pkl):
        logging.info("ULMFiT .pkl already exists, skipping export.")
        return
    # Use Fastai's built-in IMDb dataset (prestructured)
    dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5)
    learn.load(r"D:\Projects\NLP_Sentiment_Analysis\models\awd_lstm_ulmfit_imdb")
    learn.export(ulmfit_pkl)

timed_step("ULMFiT export", export_ulmfit)

# -------------------------------
# 3. Export BERT
# -------------------------------
def export_bert():
    bert_model = BertForSequenceClassification.from_pretrained(
        r"D:\Projects\NLP_Sentiment_Analysis\models\bert_model"
    )
    bert_tokenizer = BertTokenizer.from_pretrained(
        r"D:\Projects\NLP_Sentiment_Analysis\models\bert_model"
    )
    with open(r"D:\Projects\NLP_Sentiment_Analysis\models\bert_model.pkl", "wb") as f:
        pickle.dump((bert_model, bert_tokenizer), f)

timed_step("BERT export", export_bert)

# -------------------------------
# Summary
# -------------------------------
logging.info(" Export script finished. Check models folder for .pkl files.")
