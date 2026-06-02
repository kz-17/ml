# GPT

用 PyTorch 從頭實作的 GPT (Generative Pre-trained Transformer) 模型。

## 專案結構

```
GPT/
├── gpt.py        # 模型核心：GPTConfig, CausalSelfAttention, MLP, TransformerBlock, GPT
├── train.py      # 訓練與生成腳本
├── input.txt     # 訓練用文字資料（需自行準備）
└── README.md
```

## 模型架構

- **CausalSelfAttention** — 因果多頭自注意力，使用 triangular mask 確保 autoregressive 特性
- **MLP** — 兩層線性層 + GELU 激活，內維度擴張 4 倍
- **TransformerBlock** — Pre-LayerNorm + 殘差連接
- **GPT** — Token Embedding + Positional Embedding + N 個 Transformer Block + LayerNorm + LM Head

## 使用方式

```bash
pip install torch tiktoken
```

```bash
# 準備訓練資料
echo "Your training text here" > input.txt

# 訓練
python train.py

# 生成文字
python train.py generate
```

## 超參數

可在 `train.py` 中調整 `GPTConfig`：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| vocab_size | 50257 | 詞彙表大小（GPT-2） |
| n_embd | 128 | 嵌入維度 |
| n_head | 4 | 注意力頭數 |
| n_layer | 4 | Transformer 層數 |
| block_size | 64 | 最大序列長度 |
| dropout | 0.1 | Dropout 比率 |
