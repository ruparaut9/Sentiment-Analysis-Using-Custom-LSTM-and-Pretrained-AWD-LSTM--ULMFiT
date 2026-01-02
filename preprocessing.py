import torch
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd
import re
#from utils.preprocessing import build_vocab, IMDBCsvDataset
from models.custom_lstm import SentimentLSTM


# 1. Tokenizer
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove punctuation
    return text.split()

# 2. Build vocabulary
def build_vocab(csv_file, min_freq=2):
    df = pd.read_csv(csv_file)
    counter = Counter()
    for text in df["text"]:
        tokens = tokenize(text)
        counter.update(tokens)
    vocab = {"<PAD>":0, "<UNK>":1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# 3. Encode text
def encode(text, vocab, max_len=256):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# 4. Custom Dataset
class IMDBCsvDataset(Dataset):
    def __init__(self, csv_file, vocab, max_len=256):
        self.data = pd.read_csv(csv_file)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]
        ids = encode(text, self.vocab, self.max_len)
        return torch.tensor(ids), torch.tensor(label)

# ---------------------------
# Debug block for tracking
# ---------------------------
if __name__ == "__main__":
    print("Building vocabulary from training data...")
    vocab = build_vocab("data/imdb_train.csv")
    print(" Vocab size:", len(vocab))

    print(" Loading dataset...")
    dataset = IMDBCsvDataset("data/imdb_train.csv", vocab)
    print("Dataset size:", len(dataset))

    print("Checking first sample...")
    sample_x, sample_y = dataset[0]
    print("Sample tensor shape:", sample_x.shape)   # should be [256]
    print("Sample label:", sample_y.item())         # 0 or 1
    print("Sample tensor (first 20 ids):", sample_x[:20].tolist())
