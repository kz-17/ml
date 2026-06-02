# LSTM Language Model (非 Transformer 語言模型)

使用 LSTM (Long Short-Term Memory) 實現的語言模型，完全不使用 Transformer/Attention 機制。

## 模型架構

- **詞嵌入層** (Embedding): 將單詞轉為密集向量
- **LSTM 層**: 2 層 LSTM，隱藏層大小 256，處理序列依賴
- **全連接層** (Linear): 將隱藏狀態映射到詞彙表大小的輸出

```
Embedding(vocab_size, 128) → LSTM(128, 256, num_layers=2) → Linear(256, vocab_size)
```

## 檔案結構

```
├── data/
│   ├── train.txt    # 訓練語料
│   └── test.txt     # 測試語料
├── model.py         # LSTM 語言模型定義
├── train.py         # 訓練腳本
├── generate.py      # 文字生成腳本
├── checkpoints/     # 模型儲存位置（訓練後產生）
└── README.md
```

## 使用方式

### 安裝依賴

```bash
pip install torch
```

### 訓練模型

```bash
python train.py
```

訓練完成後模型會儲存在 `checkpoints/model.pt`。

### 生成文字

```bash
python generate.py --prompt "語言模型" --length 15 --temperature 0.8
```

參數說明：
- `--prompt`: 起始文字
- `--length`: 生成詞彙數量
- `--temperature`: 隨機性（越高越隨機，建議 0.5~1.2）

### 測試困惑度

```bash
python eval.py
```

## 評估指標

使用 **困惑度 (Perplexity, PPL)** 評估模型，計算方式：

```
PPL = exp(CrossEntropyLoss)
```

PPL 越低表示模型對資料的預測能力越好。
