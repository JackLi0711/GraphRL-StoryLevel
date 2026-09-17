# 2026 年以前的 server Run 一律屬於 Kyle 帳號

直到 2026 年才在 Jack 的 server 帳號下設定好可運作的 OpenSees 環境，
在那之前的所有 server 訓練都是用 Kyle 帳號下的環境跑的。
因此沒有資料夾人名可讀的 Unfiled server Run（V100 / RTX3080），
Operator 依日期推定：2026 年以前為 `Server-Kyle`，2026 年（含）以後為 `Server-Jack`。

## Consequences

這個限制無法從程式碼或 log 看出來 —— log 裡只有 GPU 型號，沒有帳號名稱，
而 Jack 與 Kyle 兩個帳號用的是同一台 V100。已有人名的 server Run 與此規則完全一致：
`Server-Kyle` 橫跨 2025_04_14 → 2025_12_29，`Server-Jack` 橫跨 2026_01_02 → 2026_06_11，沒有例外。

規則以年份為界而非逐年列舉，所以 2027 年以後的 Unfiled server Run 會歸為 `Server-Jack`。
若 Kyle 帳號在 2026 年後又被使用，這條規則需重新檢視（界線年份定義於 `build_index.py` 的 `JACK_SERVER_ACCOUNT_FROM_YEAR`）。

無法辨識的機器不推定是 server 還是 local：Operator 標為 `Unknown-{人}` 並產生警告，
應把該機器加入 `build_index.py` 的 `DEVICES` 後重新建立索引。
