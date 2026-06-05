# week1
https://github.com/kz-17/ml/tree/main/week1  
使用gemini  
對話紀錄: https://gemini.google.com/share/728da21c98e7
### 鄰居變換 (2-opt Swap)
將路徑中的一段「反轉」  
Ex.  
原始： A -> B -> ... -> C -> D    
變換後： A -> C -> ... -> B -> D  
### 局部最優解 (Local Optimum)
爬山演算法最大的限制在於它非常「貪婪」。一旦周圍的鄰居都比目前的高度低，它就會停止  
**解決方案**： 如果結果不理想，通常會搭配 隨機重新開始 (Random Restart)，或是改用 模擬退火 (Simulated Annealing)，允許在一定機率下往低處走，以跳出局部陷阱  
# Week2
https://github.com/kz-17/ml/tree/main/Week2

# week3
https://github.com/kz-17/ml/tree/main/week3  
對話紀錄: https://gemini.google.com/u/1/share/continue/f2104f0fcc59
## 使用 PyTorch 訓練多層感知器 (MLP)
1. nn.Module 與 nn.Sequential：
   - 所有的神經網路都是繼承自 nn.Module 的類別
   - nn.Sequential 可以像堆積木一樣，把線性層 (nn.Linear) 和激活函數 (nn.ReLU, nn.Sigmoid) 按順序串接起來
2. 激活函數的作用
   - ReLU()：負責打破線性關係，讓神經網路可以學習彎曲、複雜的決策邊界
   - Sigmoid()：放在最後一層，利用公式 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 將任意實數輸出壓縮到 $0$ 與 $1$ 之間
3. 優化器
   - optimizer.zero_grad()：清除上一步的舊梯度
   - loss.backward()：執行自動微分，計算所有參數的當前梯度
   - optimizer.step()：根據剛才算出的梯度，將所有參數稍微往正確的方向移動一點點
4. torch.no_grad()
   - PyTorch 停止追蹤計算圖，讓推論速度變快且不佔用多餘記憶體
# Week4
https://github.com/kz-17/ml/tree/main/HW4  
使用OpenCode
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

# Week5
https://github.com/kz-17/ml/tree/main/HW5  
使用OpenCode  
AI Agent with sandbox and manual authorization, powered by MLX local model (Qwen2.5-1.5B).

## 功能

- 基於 MLX 的本機 LLM（Qwen2.5-1.5B）
- 對話記憶與關鍵資訊提取
- **沙盒安全機制**：限制檔案存取在程式目錄內
- **Shell 指令路徑檢測**：自動分析指令中的檔案路徑，攔截外部存取
- **手動授權**：所有外部檔案存取與 shell 指令均需使用者核可

## 使用方式

```bash
python agent0.py
```

### 選項

| 參數 | 說明 |
|------|------|
| `--auto` | 自動授權所有操作（跳過確認） |
| `--workspace <路徑>` | 指定工作目錄（預設：程式所在目錄） |

### 指令

| 指令 | 說明 |
|------|------|
| `/quit` | 結束程式 |
| `/memory` | 顯示 LLM 記住的關鍵資訊 |

## 安全機制

1. **路徑解析與驗證**：`resolve_path()` / `is_path_safe()` 確保所有檔案操作都在工作區內
2. **Shell 指令掃描**：`check_shell_safety()` 靜態分析 shell 指令中的檔案路徑
3. **三層攔截**：
   - `read_file` / `write_file` 工具：自動檢查目標路徑
   - `shell` 工具：額外掃描指令參數中的外部路徑
   - 所有外部存取皆彈出授權詢問

## 測試

```bash
python test_agent0.py
```

測試項目：
- 路徑解析與邊界檢查
- Shell 指令路徑提取與安全性分析
- 檔案讀寫的沙盒強制執行
- 邊界案例（相對路徑跳脫、符號連結等）
- 授權開關功能驗證

# Week6
https://github.com/kz-17/ml/tree/main/HW6
使用OpenCode
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

# 期中
https://github.com/kz-17/ml/tree/main/midtern   
原本自己寫，檔案不見後，改用OpenCode補  

title: "基於電腦視覺之手勢辨識與物聯網控制系統"  
subtitle: "整合 MediaPipe、PyTorch 與 ESP32 之智慧控制實作"  

## 系統架構

### 整體流程

```
攝影機／手部影像
       ↓
手勢辨識 AI 模組
  OpenCV + MediaPipe
       ↓  *辨識結果 (open / fist)*
 Flask API Server
       ↓
  ESP32 控制 LED
       ↓
    實體 LED
```

### 技術棧

| 項目 | 技術 | 用途 |
|------|------|------|
| 電腦視覺 | OpenCV、MediaPipe Hands | 手部偵測與關鍵點擷取 |
| 模型框架 | PyTorch | MLP 分類模型訓練 |
| 後端伺服器 | Flask | API 介面與推論服務 |
| 微控制器 | ESP32 | LED 硬體控制 |


---

## 資料收集與前處理

### 資料收集

使用 MediaPipe Hands 模型即時偵測手部 21 個關鍵點，每個關鍵點包含 (x, y, z) 三維座標，共計 63 個特徵維度。

收集流程如下：

1. 以 Webcam 捕捉手部影像
2. MediaPipe 回傳 21 個關鍵點座標
3. 正規化處理：以手腕（第 0 點）為基準進行平移，並對 z 軸進行均值歸零
4. 將特徵向量與標籤（open / close）寫入 CSV

### 正規化函數

```python
def to_row_norm(landmarks, w, h):
    pts = np.array([[lm.x*w, lm.y*h, lm.z*w] for lm in landmarks], dtype=np.float32)
    wrist = pts[0]
    pts[:, :2] -= wrist[:2]
    pts[:, 2] -= np.mean(pts[:, 2])
    return pts.flatten().tolist()
```

### 資料集規格

- 標籤類別：`open`（張開）、`close`（握拳）
- 特徵維度：63（21 點 × 3 維）
- 每類樣本數：200～500 筆
- 儲存格式：CSV

### 資料前處理

- 合併所有 CSV 檔案為單一資料集
- 移除空值與異常樣本
- 確保各類別樣本平衡

---

## 模型設計與訓練

### 模型架構

採用多層感知器（MLP）分類器：

- 輸入層：63 個神經元（對應 63 維特徵）
- 隱藏層：兩層全連接層（含 ReLU 激活函數與 Dropout）
- 輸出層：2 個神經元（對應 open / close）

### 訓練配置

| 項目 | 設定 |
|------|------|
| 損失函數 | CrossEntropyLoss |
| 優化器 | Adam |
| 訓練週期 | 20 Epoch |
| Batch Size | 32 |

### 辨識流程

1. 載入訓練完成的模型權重
2. 即時擷取攝影機畫面
3. MediaPipe 取得手部關鍵點
4. 經正規化後輸入模型推論
5. 輸出 "Open" 或 "Close" 結果

---

## 硬體控制實作

### 控制流程

- 本機端運行 Flask API Server
- 可將辨識結果以 INSERT 寫入 MySQL/SQLite（紀錄時間）
- 同時以 POST 請求傳送至 ESP32
- ESP32 接收命令後控制 GPIO 腳位點亮對應 LED

### ESP32 通訊方式演進

### 第一版：Flask on ESP32（失敗）
- 嘗試在 ESP32 上直接運行 Flask
- 發現 ESP32 資源不足以支撐 Flask 框架

### 第二版：Requests 輪詢（效果不佳）
- 本機 Flask 作為 Server，ESP32 發送 Request 取得狀態
- 即時性不足，無法達到流暢控制

### 第三版：WebSocket（MicroPython 限制）
- 嘗試使用 WebSocket 實現雙向通訊
- 發現 MicroPython 不支援標準 websockets 函式庫
- 解決方案：改用 Arduino 搭配 uwebsockets 函式庫

### 第四版：MQTT（最終方案探索）
- PC 端作為 MQTT Broker 需要繁瑣設定（註冊、開帳號、金鑰驗證）
- Broker 需保持開啟，IP 與 Port 需正確設定
- ESP32 需與 PC 位於同一網段
- 最終採用更簡潔的 HTTP + Socket 方案

---

## 實驗結果

### 模型表現

- MLP 模型在 20 個 Epoch 內收斂
- 對「張開」與「握拳」兩種手勢具有良好的分類能力
- 即時推論延遲低（OpenCV 30fps 下可流暢運行）

### 系統整合測試

- Flask Server 成功接收手勢辨識結果
- ESP32 成功接收命令並控制 LED
- 端到端延遲在可接受範圍內（< 500ms）

### 展示影片

| 檔案 | 內容 |
|------|------|
| `VID_20250823_144629.mp4` | 手勢辨識過程側錄 |
| `ouput.mp4` | 系統運作展示 |
| `1.mp4` | ESP32 控制 LED 成果展示 |

---

## 問題檢討與解決

### 資料收集問題

- 初期樣本數量不足，導致模型泛化能力差
- 不同角度與光線下的辨識結果差異較大
- 解決方式：增加多樣化場景的資料收集，並加強前處理

### 通訊協定挑戰

- ESP32 無法運行 Flask → 改為 Arduino 韌體，讓 ESP32 作為 HTTP Client
- MicroPython 不支援標準 WebSocket →改用 Arduino + uwebsockets
- MQTT 設定繁瑣（需註冊帳號、金鑰驗證），且需固定 IP

### 系統穩定性

- Wi-Fi 連線不穩定會導致控制延遲
- 建議後續導入本地 MQTT Broker（如 Mosquitto）以提升可靠性

---

