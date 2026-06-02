import torch
import json
import argparse


def load_model():
    from model import LSTMLanguageModel
    with open('checkpoints/vocab.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    stoi = data['stoi']
    itos = {int(k): v for k, v in data['itos'].items()}
    vocab_size = len(stoi)
    model = LSTMLanguageModel(vocab_size)
    model.load_state_dict(torch.load('checkpoints/model.pt', map_location='cpu'))
    model.eval()
    return model, stoi, itos


def generate(model, stoi, itos, prompt, length=20, temperature=1.0):
    device = next(model.parameters()).device
    words = prompt.split()
    indices = [stoi.get(w, stoi['<unk>']) for w in words]
    x = torch.tensor([indices], dtype=torch.long).to(device)
    hidden = None
    generated = words.copy()

    for _ in range(length):
        with torch.no_grad():
            logits, hidden = model(x, hidden)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            if next_idx == stoi.get('<pad>', 0):
                continue
            next_word = itos.get(next_idx, '<unk>')
            generated.append(next_word)
            x = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return ' '.join(generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='語言模型')
    parser.add_argument('--length', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    model, stoi, itos = load_model()
    result = generate(model, stoi, itos, args.prompt, args.length, args.temperature)
    print(result)
