#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

my_path = "./data/tair10_SE.txt"

def main():
    # -----------------------------
    # 1. 讀取 SE 事件 CSV 並 index + prob
    # -----------------------------

    # folder = "./t_allParse_exon"
    # txt_folder = "./t_allParse_exon"

    folder = "./t_result_exon_1000"
    txt_folder = "./t_result_exon_1000"

    # se_df = pd.read_csv("./data/SE_events_0419.csv")
    se_df = pd.read_csv(my_path, sep="\t")
    se_df["index"] = -1
    se_df["prob"] = -1.0

    print("資料夾路徑：", folder)

    # 建立 gene → (index, content) 映射
    gene_to_index = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        content = open(path, encoding="utf-8").read()
        mg = re.search(r"Gene\s*=\s*(\S+)", content)
        mi = re.search(r"_g_(\d+)\.txt", fname)
        if mg and mi:
            gene_to_index[mg.group(1)] = (int(mi.group(1)), content)

    # 為每筆事件填入 index 與 prob
    for i, row in se_df.iterrows():
        gene = row["gene"]
        ss_5, ss_3 = int(row["5ss"]), int(row["3ss"])
        if gene not in gene_to_index:
            continue
        idx, text = gene_to_index[gene]
        se_df.at[i, "index"] = idx

        # 在解析檔中尋找對應的 p 值
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"^(\d+),\s*(\d+),\s*([0-9.eE+\-]+)$", line)
            if not m:
                continue
            three_val, five_val, p = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if three_val == ss_3 and five_val == ss_5:
                se_df.at[i, "prob"] = p
                break

    # 輸出有找到 prob 的事件
    os.makedirs("./SE_result", exist_ok=True)
    filtered_df = se_df[se_df["prob"] != -1.0]
    filtered_df.to_csv("./SE_result/SE_events_with_prob_only.csv", index=False)

    # 動態擷取可用 index
    available_indexes = set(filtered_df["index"].unique())
    print("所有成功對應到的 index：", sorted(available_indexes))
    print("基因個數：", len(available_indexes))
    print(f"資料筆數： {len(filtered_df)}")

    # -----------------------------
    # 2. 掃描解析結果並產生 exon records
    # -----------------------------
    # se_info = {
    #     (r["gene"], r["5ss"], r["3ss"]): r["PSI"]
    #     for _, r in pd.read_csv("./data/SE_events_0419.csv").iterrows()
    # }

    se_info = {
        (r["gene"], r["5ss"], r["3ss"]): r["PSI"]
        for _, r in pd.read_csv(my_path, sep="\t").iterrows()
    }

    records = []

    gene_re  = re.compile(r"Gene\s*=\s*(\S+)")
    idx_re   = re.compile(r"index\s*=\s*(\d+)")
    a5_re    = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    a3_re    = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    line_re  = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.eE+\-]+)\s*$")

    for fname in os.listdir(txt_folder):
        if not fname.endswith(".txt"):
            continue
        content = open(os.path.join(txt_folder, fname), encoding="utf-8").read()
        g_m = gene_re.search(content)
        i_m = idx_re.search(content)

        ann5 = [int(x) for x in re.findall(r"\d+", a5_re.search(content).group(1))] if a5_re.search(content) else []
        ann3 = [int(x) for x in re.findall(r"\d+", a3_re.search(content).group(1))] if a3_re.search(content) else []
        pair_count = min(len(ann3), len(ann5)) - 1

        for j in range(pair_count):
            three_ss = ann3[j]
            five_ss  = ann5[j+1]
            key = (g_m.group(1), five_ss, three_ss) if g_m else (None, None, None)
            classification = "SE" if key in se_info else "non-SE"
            psi_val = se_info.get(key)
            prob = None
            for line in content.splitlines():
                m = line_re.match(line.strip())
                if m and int(m.group(1)) == three_ss and int(m.group(2)) == five_ss:
                    prob = float(m.group(3))
                    break
            records.append({
                "gene": key[0],
                "5ss": five_ss,
                "3ss": three_ss,
                "PSI": psi_val,
                "index": int(i_m.group(1)) if i_m else None,
                "prob": prob,
                "classification": classification
            })

    df = pd.DataFrame(records).sort_values(by="index").reset_index(drop=True)

    # -----------------------------
    # 3. 過濾、輸出摘要、繪圖
    # -----------------------------
    df_filtered = df[df["index"].isin(available_indexes)]
    df_filtered.to_csv("./SE_result/annotated_exon_details_filtered.csv", index=False)

    valid_df = df_filtered[
        (df_filtered["classification"] != "NA") &
        (df_filtered["prob"].notnull())
    ]
    summary = valid_df.groupby("classification").prob.agg(
        count="count", avg_prob="mean", std_prob="std"
    ).reset_index()
    summary.to_csv("./SE_result/annotated_exon_summary.csv", index=False)

    # 繪圖設定
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Boxplot + Swarmplot：SE 紅色在左、non-SE 黑色在右
    sns.boxplot(
        x="classification", y="prob", data=valid_df,
        ax=ax1, order=["SE", "non-SE"], showfliers=False,
        palette={"SE": "red", "non-SE": "black"},
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"}
    )
    sns.swarmplot(
        x="classification", y="prob", data=valid_df,
        ax=ax1, order=["SE", "non-SE"],
        palette={"SE": "red", "non-SE": "black"},
        dodge=False, size=11.5, linewidth=0
    )
    ax1.set_xlabel("Classification", fontsize=40)
    ax1.set_ylabel("Exon Score", fontsize=40)
    ax1.tick_params(axis='x', labelsize=40)  # 將 x 軸刻度字型大小設為 15
    ax1.tick_params(axis='y', labelsize=40)  # 將 x 軸刻度字型大小設為 15
    ax1.set_yticks([0.0, 1.0])
    
    # 平滑 eCDF
    def kde_cdf_smooth(vals, grid_size=300):
        arr = np.array(vals)
        if arr.size == 0:
            return np.array([]), np.array([])
        x = np.linspace(arr.min(), arr.max(), grid_size)
        kde = gaussian_kde(arr)
        pdf = kde(x)
        cdf = np.cumsum(pdf) / np.sum(pdf)
        return x, cdf

    x_se,  cdf_se  = kde_cdf_smooth(valid_df[valid_df["classification"]=="SE"]["prob"].values)
    x_non, cdf_non = kde_cdf_smooth(valid_df[valid_df["classification"]=="non-SE"]["prob"].values)
    ax2.plot(x_se,  cdf_se,  label="SE",     color="red",   linewidth=10)
    ax2.plot(x_non, cdf_non, label="non-SE", color="black", linewidth=10)
    ax2.set_xlabel("Exon Score", fontsize=40)
    ax2.set_ylabel("eCDF", fontsize=40)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0.0, 1.0])
    ax2.set_yticks([0.0, 1.0])
    ax2.tick_params(axis='x', labelsize=40)  # 將 x 軸刻度字型大小設為 15
    ax2.tick_params(axis='y', labelsize=40)  # 將 x 軸刻度字型大小設為 15

    ax2.legend(
        fontsize=40,
        frameon=False,
        loc="lower right",
        handlelength=0.2,       # 線段長度設 0，就只看 marker
        scatterpoints=1,      # 如果用 scatter 的 legend 就是 1 個點
        bbox_to_anchor=(1.07, -0.13)
    )

    plt.tight_layout()
    plt.savefig("./SE_result/SE_t_k.png", dpi=300)
    plt.show()

    print("✅ Done!")

if __name__ == "__main__":
    main()
    # top_k =  {34, 75, 122, 258, 277, 308, 310, 318, 483, 526, 809, 893}
    # current = {34, 75, 122, 277, 308, 310, 318, 526, 809, 893}
    # print(top_k-current)
    # print(len(top_k-current))

    # # {258, 483}
