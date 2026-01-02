import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, output_dim=1, num_layers=2, dropout=0.5):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)              # [batch, seq_len, embed_dim]
        lstm_out, (h, c) = self.lstm(embedded)    # [batch, seq_len, hidden_dim]
        final_hidden = h[-1]                      # last layer hidden state
        out = self.fc(final_hidden)               # [batch, output_dim]
        return self.sigmoid(out).squeeze()
if __name__ == "__main__":
    # Example: vocab size = 25000
    model = SentimentLSTM(vocab_size=25000)

    # Create a dummy batch: 32 reviews, each of length 256
    dummy_input = torch.randint(0, 25000, (32, 256))

    # Forward pass
    output = model(dummy_input)

    print(" Model executed successfully")
    print("Output shape:", output.shape)   # should be [32]
    print("Output sample:", output[:5])    # first 5 predictions
