# Rollout 展示網頁重播預先計算的 Design Step 紀錄，而非即時執行

展示網頁的主要場合是口試與簡報，必須離線、零失敗地開啟。
但一個 Design Step 不只是模型前向傳播：`env.step()` 會呼叫 OpenSees（或僅限 Windows 的 PISA3D exe）做結構分析，
還需要 PyTorch 與 torch_geometric。因此 v1 在本機預先跑完 Rollout，把每個 Design Step 輸出成紀錄（JSON），
網頁只負責重播；網頁本身是單一、無外部相依的 HTML 檔，連同 JSON 一併提交進 git。

## Consequences

Design Step 紀錄是 Python 端與網頁之間的契約。Rollout 迴圈被抽成逐步產出紀錄的 generator，
既有的 matplotlib 輸出、JSON 匯出、以及未來的即時執行都是它的消費者。
v2 要在網頁上即時執行時，只需新增一個連到本機 model 與 env 的資料來源、串流同樣格式的紀錄，網頁的呈現邏輯不必重寫。
因此修改紀錄格式時必須同時顧及兩種來源；不要在網頁端直接假設「資料一次全部到齊」。

## Considered Options

- 即時執行（網頁連本機後端）：互動性最高，但簡報現場依賴 GPU、openseespy 與 exe，失敗風險最高 —— 延到 v2。
- 直接嵌入既有的 matplotlib 逐步 PNG：最快，但無法互動、檔案大，且無法沿用到即時執行。
