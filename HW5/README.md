# Agent0

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
