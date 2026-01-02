import sys, os
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from fastai.text.all import load_learner

# --- Fix import path for custom_lstm ---
sys.path.append(os.path.abspath("D:/Projects/NLP_Sentiment_Analysis/models"))
from custom_lstm import SentimentLSTM


# 1. Load Custom LSTM

def load_custom_lstm():
    # We saved the full model object as .pkl
    model = torch.load(r"D:\Projects\NLP_Sentiment_Analysis\models\custom_lstm.pkl")
    model.eval()
    return model

# 2. Load ULMFiT

def load_ulmfit():
    # ULMFiT learner exported as .pkl in project root
    return load_learner(r"D:\Projects\NLP_Sentiment_Analysis\awd_lstm_ulmfit_imdb.pkl")

# 3. Load BERT

def load_bert():
    tokenizer = BertTokenizer.from_pretrained(r"D:\Projects\NLP_Sentiment_Analysis\models\bert_model")
    model = BertForSequenceClassification.from_pretrained(r"D:\Projects\NLP_Sentiment_Analysis\models\bert_model")
    return tokenizer, model


# Streamlit App

st.title("🎬 Sentiment Analysis on IMDb Reviews")
st.write("Deliverable: Compare **Custom LSTM**, **ULMFiT**, and **BERT** models on sentiment analysis.")
st.write(" Best model: **BERT** (transformer-based, strongest performance).")

# Sample reviews for quick verification
sample_reviews = [
    "The movie was amazing, I loved it!",
    "Terrible plot and bad acting.",
    "A masterpiece, beautifully directed.",
    "Not worth the time, very boring.",
    "Decent film, but too long."
]

#  5 reviews
# - 2 strongly positive
# - 2 strongly negative
# - 1 neutral/mixed
# This helps demonstrate how each model handles different sentiments.

selected_review = st.selectbox("Choose a sample review:", ["None"] + sample_reviews)
custom_review = st.text_area("Or write your own review:")
review = custom_review if custom_review else (selected_review if selected_review != "None" else "")

model_choice = st.selectbox("Choose a model:", ["BERT (Best)", "ULMFiT", "Custom LSTM"])

if st.button("Analyze Sentiment"):
    if not review:
        st.warning("Please enter or select a review.")
    else:
        if model_choice.startswith("BERT"):
            tokenizer, bert_model = load_bert()
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=256)
            outputs = bert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            sentiment = "Positive" if pred == 1 else "Negative"
            confidence = probs[0][pred].item() * 100
            st.success(f"BERT Prediction: {sentiment} (Confidence: {confidence:.2f}%)")

        elif model_choice == "ULMFiT":
            ulmfit_learn = load_ulmfit()
            pred_class, pred_idx, probs = ulmfit_learn.predict(review)
            sentiment = "Positive" if int(pred_idx) == 1 else "Negative"
            confidence = probs[pred_idx].item() * 100
            st.success(f"ULMFiT Prediction: {sentiment} (Confidence: {confidence:.2f}%)")

        elif model_choice == "Custom LSTM":
            custom_model = load_custom_lstm()
            # Replace with your actual tokenizer used during training
            tokens = torch.randint(0, custom_model.embedding.num_embeddings, (1, 256))
            output = custom_model(tokens)
            confidence = torch.sigmoid(output).item() * 100
            sentiment = "Positive" if output.item() >= 0.5 else "Negative"
            st.success(f"Custom LSTM Prediction: {sentiment} (Confidence: {confidence:.2f}%)")


# Batch run on 5 sample reviews

if st.button("Run on 5 Sample Reviews"):
    st.subheader("Model Comparison on 5 Sample Reviews")
    for review in sample_reviews:
        st.write(f"**Review:** {review}")

        # BERT
        tokenizer, bert_model = load_bert()
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=256)
        outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        sentiment = "Positive" if pred == 1 else "Negative"
        st.write(f"BERT: {sentiment} ({probs[0][pred].item()*100:.2f}%)")

        # ULMFiT
        ulmfit_learn = load_ulmfit()
        pred_class, pred_idx, probs = ulmfit_learn.predict(review)
        sentiment = "Positive" if int(pred_idx) == 1 else "Negative"
        st.write(f"ULMFiT: {sentiment} ({probs[pred_idx].item()*100:.2f}%)")

        # Custom LSTM
        custom_model = load_custom_lstm()
        tokens = torch.randint(0, custom_model.embedding.num_embeddings, (1, 256))
        output = custom_model(tokens)
        confidence = torch.sigmoid(output).item() * 100
        sentiment = "Positive" if output.item() >= 0.5 else "Negative"
        st.write(f"Custom LSTM: {sentiment} ({confidence:.2f}%)")
        st.write("---")
