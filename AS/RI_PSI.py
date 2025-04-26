import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# 讀入資料
df = pd.read_csv("RI_result/annotated_intron_details_filtered.csv")

# 過濾 RI 資料
ri_df = df[df["classification"] == "RI"]

# # 抓出 PSI 與 prob 欄位並丟掉 NaN/Inf
# ri_df = ri_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["PSI", "prob"])

# 清理前筆數
before_count = len(ri_df)

# 抓出 PSI 與 prob 欄位並丟掉 NaN/Inf
ri_df = ri_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["PSI", "prob"])

# 清理後筆數
after_count = len(ri_df)
removed_count = before_count - after_count

print(f"Total RI entries before filtering: {before_count}")
print(f"Entries removed due to NaN/Inf in PSI or prob: {removed_count}")
print(f"Remaining entries after filtering: {after_count}")

# 提取數值
psi = ri_df["PSI"].astype(float)
prob = ri_df["prob"].astype(float)

# 計算 Pearson correlation
corr, pval = pearsonr(psi, prob)
print(f"Pearson correlation: {corr:.4f}, p-value: {pval:.4e}")

# 畫圖
plt.figure(figsize=(6, 5))
sns.scatterplot(x=psi, y=prob, alpha=0.6, edgecolor=None)
plt.title(f"Pearson correlation r = {corr:.2f}*** (p-value = {pval:.2e})")
plt.xlabel("PSI")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("PSI_vs_score_RI.png", dpi=300)
