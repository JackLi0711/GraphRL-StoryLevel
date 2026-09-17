# 以 best 而非 last 作為 Run 的代表指標

一個 Run 的 `testing_record.txt` 是逐測試點的陣列（長度 20 至 1200 不等），
必須挑一個數字代表整個 Run 才能排序與比較。我們選 **best**（最佳 Saving Ratio），
因為 `model_HighestScore.pt` 正是實際會拿去做 inference 的權重，best 有操作意義。

## Considered Options

取 **last**（最後一個測試點）是更直覺的選擇，也是多數 RL 論文的慣例。
我們沒有採用，因為在現有 81 個 Run 上，兩種取法會排出不同的名次：
best 與 last 的差距平均 0.100、最大 0.380，前五名會換人。

## Consequences

單看 best 會把訓練不穩定性藏起來，因此 **best 必須永遠與 last 及
Convergence Gap 一起呈現**，gap > 0.1 標記為未收斂。這個 gap 本身帶有資訊：
初步觀察 DQN 系的 gap 在 0.003~0.025，OC 系可達 0.341，
亦即「OC 訓練較不穩定」是可量化的，只取 best 會看不到。
