# week3
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
