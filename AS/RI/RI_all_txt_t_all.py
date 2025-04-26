#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

my_path = "./data/tair10_RI.txt"

def main():
    # -----------------------------
    # 1. 讀取 RI 事件 CSV 並 index + prob
    # -----------------------------

    folder = "./t_allParse_intron"
    txt_folder = "./t_allParse_intron"

    # folder = "./t_result_intron_1000"
    # txt_folder = "./t_result_intron_1000"   
    

    ri_df = pd.read_csv(my_path, sep="\t")
    ri_df["index"] = -1
    ri_df["prob"] = -1.0

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

    # 填入 index 與 prob
    line_re = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.eE+\-]+)\s*$")
    for i, row in ri_df.iterrows():
        gene = row["gene"]
        s5, s3 = int(row["5ss"]), int(row["3ss"])
        if gene not in gene_to_index:
            continue
        
        idx, text = gene_to_index[gene]
        ri_df.at[i, "index"] = idx

        for line in text.splitlines():
            m = line_re.match(line)
            if not m:
                continue
            five_val, three_val, p = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if five_val == s5 and three_val == s3:
                ri_df.at[i, "prob"] = p
                break

    # 輸出找到 prob 的事件
    os.makedirs("./RI_result", exist_ok=True)
    filtered_df = ri_df[ri_df["prob"] != -1.0]
    filtered_df.to_csv("./RI_result/RI_events_with_prob_only.csv", index=False)

    # 動態擷取可用 index
    available_indexes = set(filtered_df["index"].unique())
    print("所有成功對應到的 index：", sorted(available_indexes))
    print("基因個數：", len(available_indexes))
    print(f"資料筆數： {len(filtered_df)}")

    # -----------------------------
    # 2. 掃描解析結果並產生 intron records
    # -----------------------------
    ri_info = {
        (r["gene"], r["5ss"], r["3ss"]): r["PSI"]
        for _, r in pd.read_csv(my_path, sep="\t").iterrows()
    }

    records = []

    gene_re    = re.compile(r"Gene\s*=\s*(\S+)")
    idx_re     = re.compile(r"index\s*=\s*(\d+)")
    ann5_re    = re.compile(r"Annotated\s+5SS:\s*\[([^\]]*)\]")
    ann3_re    = re.compile(r"Annotated\s+3SS:\s*\[([^\]]*)\]")

    for fname in os.listdir(txt_folder):
        if not fname.endswith(".txt"):
            continue
        content = open(os.path.join(txt_folder, fname), encoding="utf-8").read()
        g_m = gene_re.search(content)
        i_m = idx_re.search(content)
        ann5 = [int(x) for x in re.findall(r"\d+", ann5_re.search(content).group(1))] if ann5_re.search(content) else []
        ann3 = [int(x) for x in re.findall(r"\d+", ann3_re.search(content).group(1))] if ann3_re.search(content) else []
        pair_count = min(len(ann5), len(ann3))

        for j in range(pair_count):
            five_ss, three_ss = ann5[j], ann3[j]
            key = (g_m.group(1), five_ss, three_ss) if g_m else (None, None, None)
            classification = "RI" if key in ri_info else "non-RI"
            psi_val = ri_info.get(key)
            prob = None
            for line in content.splitlines():
                m = line_re.match(line.strip())
                if m and int(m.group(1)) == five_ss and int(m.group(2)) == three_ss:
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

    df = pd.DataFrame(records).sort_values("index").reset_index(drop=True)

    # -----------------------------
    # 3. 過濾、輸出摘要、繪圖
    # -----------------------------
    df_filtered = df[df["index"].isin(available_indexes)]
    df_filtered.to_csv("./RI_result/annotated_intron_details_filtered.csv", index=False)

    valid_df = df_filtered[
        (df_filtered["classification"] != "NA") &
        (df_filtered["prob"].notnull())
    ]
    summary = valid_df.groupby("classification")["prob"].agg(
        count="count", avg_prob="mean", std_prob="std"
    ).reset_index()
    summary.to_csv("./RI_result/annotated_intron_summary.csv", index=False)

    # 繪圖設定
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Boxplot + Swarmplot：RI 紅色在左、non-RI 黑色在右
    sns.boxplot(
        x="classification", y="prob", data=valid_df,
        ax=ax1, order=["RI", "non-RI"], showfliers=False,
        palette={"RI": "red", "non-RI": "black"},
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"}
    )
    sns.swarmplot(
        x="classification", y="prob", data=valid_df,
        ax=ax1, order=["RI", "non-RI"],
        palette={"RI": "red", "non-RI": "black"},
        dodge=False, size=11.5, linewidth=0
    )
    ax1.set_xlabel("Classification", fontsize=40)
    ax1.set_ylabel("Intron Score", fontsize=40)
    ax1.set_yticks([0.0, 1.0])
    ax1.tick_params(axis='x', labelsize=40)  # 將 x 軸刻度字型大小設為 15
    ax1.tick_params(axis='y', labelsize=40)  # 將 x 軸刻度字型大小設為 15

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

    x_ri, cdf_ri     = kde_cdf_smooth(valid_df[valid_df["classification"]=="RI"]["prob"])
    x_nonri, cdf_nonri = kde_cdf_smooth(valid_df[valid_df["classification"]=="non-RI"]["prob"])
    ax2.plot(x_ri, cdf_ri, label="RI", color="red", linewidth=10)
    ax2.plot(x_nonri, cdf_nonri, label="non-RI", color="black", linewidth=10)
    ax2.set_xlabel("Intron Score", fontsize=40)
    ax2.set_ylabel("eCDF", fontsize=40)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0.0, 1.0])
    ax2.set_yticks([0.0, 1.0])

    ax2.tick_params(axis='x', labelsize=40)  # 將 x 軸刻度字型大小設為 15
    ax2.tick_params(axis='y', labelsize=40)  # 將 x 軸刻度字型大小設為 15
    # ax2.legend(fontsize=40, frameon=False, loc="lower right")

# 在 legend 裡，只顯示 marker
    ax2.legend(
        fontsize=40,
        frameon=False,
        loc="lower right",
        handlelength=0.2,       # 線段長度設 0，就只看 marker
        scatterpoints=1,      # 如果用 scatter 的 legend 就是 1 個點
        bbox_to_anchor=(1.07, -0.12)
    )

    plt.tight_layout()
    plt.savefig("./RI_result/RI_t_all.png", dpi=300)
    plt.show()

    print("✅ Done!")

if __name__ == "__main__":
    main()
    # top_k =  {1, 3, 5, 6, 7, 9, 29, 31, 32, 34, 44, 51, 55, 58, 66, 70, 75, 79, 102, 105, 112, 122, 125, 129, 134, 138, 142, 150, 152, 154, 164, 165, 168, 171, 173, 179, 180, 181, 187, 194, 205, 209, 211, 219, 223, 225, 226, 231, 251, 258, 263, 265, 268, 269, 271, 272, 276, 277, 278, 283, 287, 292, 294, 308, 309, 310, 317, 319, 323, 328, 336, 337, 338, 342, 343, 344, 355, 366, 368, 378, 384, 387, 392, 398, 399, 409, 411, 414, 415, 419, 426, 432, 437, 438, 439, 457, 473, 478, 483, 496, 500, 502, 508, 514, 521, 523, 526, 527, 529, 531, 532, 533, 551, 554, 555, 557, 559, 565, 567, 568, 569, 577, 583, 584, 586, 588, 591, 593, 594, 595, 596, 597, 598, 603, 609, 620, 629, 631, 634, 637, 650, 660, 664, 686, 689, 692, 695, 699, 701, 705, 721, 724, 727, 728, 745, 747, 749, 753, 764, 773, 776, 777, 781, 801, 809, 816, 826, 833, 834, 836, 846, 847, 857, 858, 863, 890, 899, 903, 906, 908, 912, 927, 935, 936, 943, 953, 960, 968, 970, 976, 988, 990, 994, 995, 997, 998, 1010, 1012, 1016, 1017, 1021, 1022, 1023, 1024, 1030, 1042, 1045, 1047, 1048, 1050, 1058, 1061, 1071, 1077, 1083, 1087, 1091, 1094, 1098, 1109}
    # current = {1, 3, 5, 6, 7, 9, 29, 31, 32, 44, 51, 55, 58, 66, 70, 75, 79, 105, 125, 129, 134, 138, 142, 152, 154, 164, 165, 168, 171, 173, 179, 180, 181, 194, 205, 209, 211, 219, 225, 226, 231, 251, 263, 265, 268, 269, 271, 272, 276, 277, 278, 283, 287, 292, 294, 308, 309, 310, 317, 319, 323, 328, 337, 338, 342, 344, 355, 366, 368, 378, 384, 387, 392, 398, 409, 411, 414, 419, 426, 432, 437, 438, 473, 478, 496, 500, 502, 508, 514, 521, 523, 526, 527, 529, 531, 532, 533, 551, 554, 555, 557, 559, 565, 567, 568, 569, 577, 583, 584, 586, 588, 593, 594, 595, 596, 597, 598, 609, 620, 631, 634, 637, 660, 689, 692, 695, 699, 701, 721, 724, 727, 728, 745, 747, 749, 753, 764, 773, 776, 777, 801, 809, 816, 826, 833, 834, 846, 847, 858, 863, 893, 899, 906, 908, 927, 935, 936, 943, 953, 960, 968, 970, 976, 988, 994, 995, 997, 998, 1010, 1012, 1016, 1017, 1022, 1023, 1024, 1030, 1042, 1045, 1047, 1048, 1050, 1058, 1061, 1071, 1077, 1083, 1087, 1091, 1094, 1098, 1109}
    # print(top_k-current)
    # print(len(top_k-current))

    # # {258, 903, 650, 781, 399, 912, 150, 664, 415, 34, 686, 439, 187, 705, 836, 457, 591, 336, 343, 857, 603, 990, 223, 483, 102, 112, 629, 122, 1021, 890}
