import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import os


class TextDataset(Dataset):
    def __init__(self, filepath, seq_len=10):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        words = text.split()
        vocab = Counter(words)
        self.stoi = {w: i+2 for i, (w, _) in enumerate(vocab.most_common())}
        self.stoi['<pad>'] = 0
        self.stoi['<unk>'] = 1
        self.itos = {i: w for w, i in self.stoi.items()}
        data = [self.stoi.get(w, 1) for w in words]
        self.seq_len = seq_len
        self.samples = []
        for i in range(0, len(data) - seq_len, seq_len):
            x = data[i:i+seq_len]
            y = data[i+1:i+seq_len+1]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    seq_len = 10
    epochs = 50

    dataset = TextDataset('data/train.txt', seq_len)
    if len(dataset) == 0:
        print("Not enough data. Padding with dummy sequence.")
        dataset.samples.append((
            [dataset.stoi['<pad>']] * seq_len,
            [dataset.stoi['<pad>']] * seq_len
        ))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = len(dataset.stoi)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(dataset)}")

    from model import LSTMLanguageModel
    model = LSTMLanguageModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")

    torch.save(model.state_dict(), 'checkpoints/model.pt')
    with open('checkpoints/vocab.json', 'w', encoding='utf-8') as f:
        json.dump({'stoi': dataset.stoi, 'itos': {str(k): v for k, v in dataset.itos.items()}}, f, ensure_ascii=False)
    print("Model saved to checkpoints/model.pt")


if __name__ == '__main__':
    train()
