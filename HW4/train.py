import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
from gpt import GPT, GPTConfig


class TextDataset(Dataset):
    def __init__(self, text, block_size):
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, i):
        x = self.tokens[i : i + self.block_size]
        y = self.tokens[i + 1 : i + self.block_size + 1]
        return x, y


@torch.no_grad()
def estimate_loss(model, dataloader, device):
    model.eval()
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    config = GPTConfig(
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=4,
        block_size=64,
        dropout=0.1,
    )
    model = GPT(config).to(device)

    dataset = TextDataset(text, config.block_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(1000):
        x, y = next(iter(dataloader))
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            val_loss = estimate_loss(model, dataloader, device)
            print(f"step {step:>4d} | train loss {loss.item():.4f} | val loss {val_loss:.4f}")

    torch.save(model.state_dict(), "gpt_model.pt")
    print("Model saved to gpt_model.pt")


def generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=50257,
        n_embd=128,
        n_head=4,
        n_layer=4,
        block_size=64,
    )
    model = GPT(config).to(device)
    model.load_state_dict(torch.load("gpt_model.pt", map_location=device))
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt = "Once upon a time"
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=40)
    generated_text = enc.decode(output[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated:\n{generated_text}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate()
    else:
        train()
