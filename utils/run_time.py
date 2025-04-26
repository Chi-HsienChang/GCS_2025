import matplotlib.pyplot as plt

# k 值與對應的執行時間（秒）——更新後數據

# Human index = 0
# k_values = [100, 500, 1000]
# exon_time   = [4.20, 19.98, 40.12]
# intron_time = [3.65, 20.38, 41.24]
# ss_time     = [3.65, 19.71, 40.45]

# Human index = 5
k_values = [100, 500, 1000]
exon_time   = [15.44, 81.21, 152.48]
intron_time = [15.23, 81.94, 152.69]
ss_time     = [15.45, 83.07, 150.31]

plt.figure(figsize=(8, 4.5))

# 各條線的 marker 樣式
plt.plot(k_values, ss_time, marker='o', markersize=12, linewidth=2, label="Splice Site Score")
plt.plot(k_values, exon_time, marker='p', linestyle='--', markersize=10, linewidth=2, label="Exon Score")
plt.plot(k_values, intron_time, marker='s', linestyle='-.', markersize=8, linewidth=2, label="Intron Score")

# 字體大小與標籤
plt.xticks([100, 500, 1000], labels=["100", "500", "1000"], fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("$k$", fontsize=20)
plt.ylabel("Execution Time (seconds)", fontsize=18)

# 圖例與網格
plt.legend(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 儲存圖檔
plt.savefig("Human_time.png", dpi=300)
plt.show()
