import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 準備資料 (Data Preparation)
# ==========================================
# 產生 1000 個帶有雜訊的非線性資料點 (兩個交錯的半月形)
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)

# 正規化資料 (讓特徵範圍縮放到平均值 0，標準差 1，有助於加速訓練)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 將 Numpy 陣列轉換為 PyTorch 的 Tensor 格式
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1) # 調整維度為 (N, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# ==========================================
# 2. 定義神經網路模型 (Model Definition)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 網路架構：輸入層(2) -> 隱藏層(16) -> 隱藏層(8) -> 輸出層(1)
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),        # 非線性激活函數
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()      # 將輸出壓縮到 0~1 之間，代表機率
        )

    def forward(self, x):
        return self.network(x)

model = SimpleMLP()

# ==========================================
# 3. 設定損失函數與優化器 (Loss & Optimizer)
# ==========================================
# 二元交叉熵損失 (Binary Cross Entropy Loss)，專門用於二元分類
criterion = nn.BCELoss() 
# Adam 優化器，學習率設為 0.01
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 4. 開始訓練 (Training Loop)
# ==========================================
epochs = 100
for epoch in range(epochs):
    # a. 模式切換為訓練模式
    model.train()
    
    # b. 梯度清零 (PyTorch 預設會累加梯度，每回合必須清零)
    optimizer.zero_grad()
    
    # c. 前向傳播 (Forward Pass)
    outputs = model(X_train)
    
    # d. 計算損失 (Loss)
    loss = criterion(outputs, y_train)
    
    # e. 反向傳播 (Backward Pass)
    loss.backward()
    
    # f. 更新權重 (Update Weights)
    optimizer.step()
    
    # 每 20 回合印出一次進度
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ==========================================
# 5. 模型評估 (Evaluation)
# ==========================================
model.eval() # 切換為評估模式 (關閉 Dropout / BatchNorm 等機制)
with torch.no_grad(): # 評估時不需要計算梯度，節省記憶體與算力
    test_outputs = model(X_test)
    # 若機率 >= 0.5 則預測為類別 1，否則為類別 0
    predictions = (test_outputs >= 0.5).float()
    
    # 計算準確率
    correct = (predictions == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print(f'\n測試集準確率 (Accuracy): {accuracy * 100:.2f}%')
