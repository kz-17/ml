---
title: "基於電腦視覺之手勢辨識與物聯網控制系統"
subtitle: "整合 MediaPipe、PyTorch 與 ESP32 之智慧控制實作"
---

# 摘要

本專案實作一套基於電腦視覺的手勢辨識控制系統，利用 OpenCV 與 MediaPipe 進行即時手部關鍵點偵測，並以 PyTorch 建構多層感知器（MLP）分類模型辨識「張開」與「握拳」兩種手勢。辨識結果透過 Flask API 即時傳送至 ESP32 微控制器，實現無線控制 LED 燈號切換。本研究記錄了從資料收集、模型訓練、系統整合到硬體控制之完整開發歷程，並探討了 Flask、WebSocket、MQTT 等多種通訊協定在 ESP32 上的適用性。

---

# 1. 研究動機

隨著物聯網（IoT）與邊緣運算技術的快速發展，以非接觸式手勢進行人機互動已成為智慧生活的重要方向。傳統的遙控器與觸控介面在特定場景下存在衛生、便利性及無障礙等限制。本專案旨在開發一套低成本的即時手勢辨識系統，並將其與物聯網裝置整合，實現直覺化的控制體驗。

---

# 2. 系統架構

## 2.1 整體流程

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

## 2.2 技術棧

| 項目 | 技術 | 用途 |
|------|------|------|
| 電腦視覺 | OpenCV、MediaPipe Hands | 手部偵測與關鍵點擷取 |
| 模型框架 | PyTorch | MLP 分類模型訓練 |
| 後端伺服器 | Flask | API 介面與推論服務 |
| 微控制器 | ESP32 | LED 硬體控制 |


---

# 3. 資料收集與前處理

## 3.1 資料收集

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

## 3.2 資料集規格

- 標籤類別：`open`（張開）、`close`（握拳）
- 特徵維度：63（21 點 × 3 維）
- 每類樣本數：200～500 筆
- 儲存格式：CSV

## 3.3 資料前處理

- 合併所有 CSV 檔案為單一資料集
- 移除空值與異常樣本
- 確保各類別樣本平衡

---

# 4. 模型設計與訓練

## 4.1 模型架構

採用多層感知器（MLP）分類器：

- 輸入層：63 個神經元（對應 63 維特徵）
- 隱藏層：兩層全連接層（含 ReLU 激活函數與 Dropout）
- 輸出層：2 個神經元（對應 open / close）

## 4.2 訓練配置

| 項目 | 設定 |
|------|------|
| 損失函數 | CrossEntropyLoss |
| 優化器 | Adam |
| 訓練週期 | 20 Epoch |
| Batch Size | 32 |

## 4.3 辨識流程

1. 載入訓練完成的模型權重
2. 即時擷取攝影機畫面
3. MediaPipe 取得手部關鍵點
4. 經正規化後輸入模型推論
5. 輸出 "Open" 或 "Close" 結果

---

# 5. 硬體控制實作

## 5.1 控制流程

- 本機端運行 Flask API Server
- 可將辨識結果以 INSERT 寫入 MySQL/SQLite（紀錄時間）
- 同時以 POST 請求傳送至 ESP32
- ESP32 接收命令後控制 GPIO 腳位點亮對應 LED

## 5.2 ESP32 通訊方式演進

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

# 6. 實驗結果

## 6.1 模型表現

- MLP 模型在 20 個 Epoch 內收斂
- 對「張開」與「握拳」兩種手勢具有良好的分類能力
- 即時推論延遲低（OpenCV 30fps 下可流暢運行）

## 6.2 系統整合測試

- Flask Server 成功接收手勢辨識結果
- ESP32 成功接收命令並控制 LED
- 端到端延遲在可接受範圍內（< 500ms）

## 6.3 展示影片

| 檔案 | 內容 |
|------|------|
| `VID_20250823_144629.mp4` | 手勢辨識過程側錄 |
| `ouput.mp4` | 系統運作展示 |
| `1.mp4` | ESP32 控制 LED 成果展示 |

---

# 7. 問題檢討與解決

## 7.1 資料收集問題

- 初期樣本數量不足，導致模型泛化能力差
- 不同角度與光線下的辨識結果差異較大
- 解決方式：增加多樣化場景的資料收集，並加強前處理

## 7.2 通訊協定挑戰

- ESP32 無法運行 Flask → 改為 Arduino 韌體，讓 ESP32 作為 HTTP Client
- MicroPython 不支援標準 WebSocket →改用 Arduino + uwebsockets
- MQTT 設定繁瑣（需註冊帳號、金鑰驗證），且需固定 IP

## 7.3 系統穩定性

- Wi-Fi 連線不穩定會導致控制延遲
- 建議後續導入本地 MQTT Broker（如 Mosquitto）以提升可靠性

---

