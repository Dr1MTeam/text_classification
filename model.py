import torch
import torch.nn as nn

# Одномерная сверточная нейросеть (1D-CNN)
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x, length):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        conv_outs = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = self.fc(x)
        return x

# Рекуррентная нейросеть с LSTM
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, bidirectional=False):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self,x, lengths):

        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),batch_first=True, enforce_sorted=False)

        _, (hidden, _) = self.lstm(x)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        x = self.fc(hidden)
        return x

# Рекуррентная нейросеть с GRU
class TextGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, bidirectional=False):
        super(TextGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),batch_first=True, enforce_sorted=False)

        _, hidden = self.gru(x)
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_dim * 2)
        else:
            hidden = hidden[-1]  # (batch, hidden_dim)
        x = self.fc(hidden)
        return x

# Пример использования
if __name__ == "__main__":
    vocab_size = 5000
    embed_dim = 128
    num_classes = 10
    seq_length = 50
    batch_size = 32

    # TextCNN
    model_cnn = TextCNN(vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100)
    input_data = torch.randint(0, vocab_size, (batch_size, seq_length))
    output_cnn = model_cnn(input_data,  torch.tensor([seq_length] * batch_size))
    print("TextCNN output shape:", output_cnn.shape)

    # TextLSTM
    hidden_dim = 128
    model_lstm = TextLSTM(vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, bidirectional=True)
    output_lstm = model_lstm(input_data,  torch.tensor([seq_length] * batch_size))
    print("TextLSTM output shape:", output_lstm.shape)

    # TextGRU
    model_gru = TextGRU(vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, bidirectional=True)
    output_gru = model_gru(input_data, torch.tensor([seq_length] * batch_size))
    print("TextGRU output shape:", output_gru.shape)