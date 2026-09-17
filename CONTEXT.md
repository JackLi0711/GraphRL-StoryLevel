# GraphRL Story-Level Design

以圖神經網路搭配強化學習進行鋼構架樓層級斷面設計：從過大的初始斷面出發，
逐步降級各樓層構件群組，在滿足規範檢核的前提下最小化材料用量。

## Language

### 設計問題

**Testing Geometry**:
所有測試共用的固定結構：4×4 跨、6 層、跨距 6000/8000 mm、樓高 3200 mm。
訓練用的結構是隨機取樣的，測試用的不是。
_Avoid_: test case, benchmark, 測試結構

**Saving Ratio**:
單一測試點的節材比例，`(initial_volume - final_volume) / initial_volume`。
與獎勵函數的設計無關，因此可跨 Run 比較。
_Avoid_: score, reward, performance

### 訓練紀錄

**Run**:
一次完整的訓練執行。其身分是檔案系統路徑；訓練參數中記下的輸出路徑只是訓練當下的位置，通常已經過時（見 ADR-0002）。
日常用語與工具名稱中的「checkpoint」就是指 Run（例如 checkpoint index）；但 Run 資料夾裡的模型權重檔（.pt）不是 Run。
_Avoid_: experiment, model, trial

**Filing**:
訓練結束後，人工把 Run 從預設輸出目錄搬進分類資料夾的動作。
未經此動作的 Run 稱為 Unfiled，不代表它比較舊或比較不重要。
_Avoid_: archiving, sorting, 整理

**Purpose**:
一個 Run 的意圖，由所在資料夾決定：
Experiment（調參的 trial-and-error）、Reproduction（benchmark）、Unfiled（尚未 Filing）。
_Avoid_: category, type, tag

**Operator**:
執行該 Run 的環境，格式為「地點-人」。在 V100 或 RTX3080 上訓練者為 Server-{人}，
人名取自資料夾後綴（無後綴者為 Jack）；Unfiled 者依年份推定，2026 年以前為 Kyle、2026 年（含）以後為 Jack（見 ADR-0003）。
在 RTX4080 上訓練者一律為 Local-Jack；無法辨識的機器為 Unknown-{人}。
_Avoid_: author, owner, user

**Note**:
事後對 Run 加上的附註與星號（0–3，表示重要性），與 Run 本身的資料分開保存。
與訓練當下寫下的 comment 是不同的東西：comment 描述「這次訓練做了什麼設定」，Note 記錄「看完結果之後的判斷」。
_Avoid_: comment, remark, 註解

**Convergence Gap**:
一個 Run 的最佳 Saving Ratio 減去最後一個測試點的 Saving Ratio。
用以量測訓練穩定度；大於 0.1 視為未收斂（見 ADR-0001）。
_Avoid_: overfitting, variance
