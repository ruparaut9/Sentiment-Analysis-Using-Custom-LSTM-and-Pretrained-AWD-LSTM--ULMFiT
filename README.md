# Sentiment Analysis: LSTM vs AWD-LSTM (ULMFiT) vs BERT

##  Overview
This project compares three deep learning approaches for sentiment analysis on the IMDb dataset:
- **Custom LSTM** (trained from scratch)
- **AWD-LSTM (ULMFiT)** (pretrained + fine-tuned)
- **BERT** (Transformer-based)

The objective is to evaluate how pretrained language representations improve classification accuracy and convergence compared to training an LSTM without prior linguistic knowledge.

---

## Skills Takeaway
- Sequence modeling using Recurrent Neural Networks (RNNs)
- Implementation of LSTM in PyTorch
- Understanding transfer learning in NLP
- Fine-tuning pretrained language models
- Performance comparison of deep learning architectures
- Model evaluation and error analysis

---

##  Domain
Natural Language Processing (NLP) / Deep Learning

---

##  Problem Statement
Design and implement a sentiment analysis system using a custom LSTM model trained from scratch and compare its performance with a pretrained AWD-LSTM (ULMFiT) model. Extend the system with a BERT-based transformer model to benchmark against state-of-the-art architectures.

---

##  Business Use Cases
- Customer sentiment analysis for product reviews
- Social media opinion mining
- Brand reputation monitoring
- Feedback analysis for e-commerce platforms
- Call-center transcript sentiment classification

---

##  Approach
1. Load the IMDb movie reviews dataset.
2. Preprocess text data using tokenization and padding.
3. Build and train a custom LSTM model in PyTorch.
4. Fine-tune a pretrained AWD-LSTM model using ULMFiT methodology.
5. Fine-tune a pretrained BERT model for sentiment classification.
6. Evaluate all models using identical metrics.
7. Perform comparative analysis on accuracy, convergence, and generalization.

---

##  Results
- **Custom LSTM:** Learns task-specific features but converges slowly (~82–85% accuracy).
- **AWD-LSTM (ULMFiT):** Achieves higher accuracy (~88–90%) with fewer epochs.
- **BERT:** Achieves the highest accuracy (~92–94%) with faster convergence but higher computational cost.

---

##  Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Training loss and validation loss
- Epochs to convergence



