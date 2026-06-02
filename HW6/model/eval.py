import torch
import torch.nn as nn
import json
from train import TextDataset


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('checkpoints/vocab.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    stoi = data['stoi']
    vocab_size = len(stoi)

    from model import LSTMLanguageModel
    model = LSTMLanguageModel(vocab_size).to(device)
    model.load_state_dict(torch.load('checkpoints/model.pt', map_location=device))
    model.eval()

    dataset = TextDataset('data/test.txt', seq_len=10)
    if len(dataset) == 0:
        print("Test set too small.")
        return

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"Test Loss: {avg_loss:.4f} | Test Perplexity: {ppl:.2f}")


if __name__ == '__main__':
    evaluate()
