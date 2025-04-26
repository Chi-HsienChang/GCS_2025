
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

def main():
    # 1. Read A5 events and initialize
    folder = "t_allParse_ss"
    print(f"Folder path: {folder}")
    a5_df = pd.read_csv("./data/tair10_A5.txt", sep="\t")
    a5_df["index"]    = -1
    a5_df["prob_5ss"] = -1.0
    a5_df["prob_3ss"] = -1.0

    # 2. Compile regex
    sec5_re = re.compile(
        r"Sorted 5' Splice Sites \(High to Low Probability\):([\s\S]+?)Sorted 3' Splice Sites",
        re.MULTILINE
    )
    sec3_re = re.compile(
        r"Sorted 3' Splice Sites \(High to Low Probability\):([\s\S]+)",
        re.MULTILINE
    )
    pos_re = re.compile(r"Position\s*(\d+):\s*([0-9.eE+\-]+)")

    # 3. Build mapping gene -> (index, text)
    gene_to_index = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        text = open(os.path.join(folder, fname), encoding="utf-8").read()
        mg = re.search(r"Gene\s*=\s*(\S+)", text)
        mi = re.search(r"_g_(\d+)\.txt", fname)
        if mg and mi:
            gene_to_index[mg.group(1)] = (int(mi.group(1)), text)

    # 4. Extract probabilities
    for i, row in a5_df.iterrows():
        gene = row["gene"]
        ss5  = int(row["5ss_alternative"])
        ss3  = int(row["3ss_pairing"])
        if gene not in gene_to_index:
            continue

        idx, text = gene_to_index[gene]
        a5_df.at[i, "index"] = idx

        # 5'
        prob5 = -1.0
        m5 = sec5_re.search(text)
        if m5:
            for line in m5.group(1).splitlines():
                m = pos_re.match(line.strip())
                if m and int(m.group(1)) == ss5:
                    prob5 = float(m.group(2))
                    break

        # 3'
        prob3 = -1.0
        m3 = sec3_re.search(text)
        if m3:
            for line in m3.group(1).splitlines():
                m = pos_re.match(line.strip())
                if m and int(m.group(1)) == ss3:
                    prob3 = float(m.group(2))
                    break

        a5_df.at[i, "prob_5ss"] = prob5
        a5_df.at[i, "prob_3ss"] = prob3

    # 5. Save & filter
    os.makedirs("./A5_result", exist_ok=True)
    a5_df.to_csv("./A5_result/A5_events_with_prob.csv", index=False)
    filtered = a5_df[(a5_df["prob_5ss"] > 0) & (a5_df["prob_3ss"] > 0)]
    # filtered["PSI"] /= 100
    filtered.to_csv("./A5_result/A5_events_with_prob_only.csv", index=False)

    indexes = sorted(filtered["index"].unique())
    print("Matched indexes:", indexes)
    print("Number of genes:", len(indexes))
    print("Number of events:", len(filtered))
    # 6. Get max/min PSI rows for each index
    # indexes = sorted(filtered["index"].unique())
    records = []
    for idx in indexes:
        sub = filtered[filtered["index"] == idx]
        gene = sub["gene"].iloc[0]
        max_row = sub.loc[sub["PSI"].idxmax()]
        min_row = sub.loc[sub["PSI"].idxmin()]
        records.append({
            "index":    idx,
            "gene":     gene,
            "PSI":      max_row["PSI"],
            "prob_5ss": max_row["prob_5ss"],
            "type":     "major"
        })
        records.append({
            "index":    idx,
            "gene":     gene,
            "PSI":      min_row["PSI"],
            "prob_5ss": min_row["prob_5ss"],
            "type":     "minor"
        })
    df_extremes = pd.DataFrame(records)

    # 7. Drop cases where scores are identical
    df_extremes = df_extremes[
        df_extremes.groupby("index")["prob_5ss"]
                   .transform(lambda x: x.nunique() > 1)
    ]

    stats_df = (
        df_extremes
        .groupby("type")["prob_5ss"]
        .agg(count="count", avg_prob="mean", std_prob="std")
        .reset_index()
    )

    print("Statistics:")
    print(stats_df)

    
    # 8 & 9. Plot: left = transparent box + colored points; right = scatter + Pearson
    r, p = pearsonr(df_extremes["PSI"], df_extremes["prob_5ss"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={"width_ratios": [1, 1.2]}
    )

    # --- Left: transparent boxplot + colored swarm ---
    # sns.boxplot(
    #     x="type", y="prob_5ss", data=df_extremes,
    #     order=["minor", "major"],
    #     showfliers=False,
    #     boxprops=dict(facecolor="none", edgecolor="black"),
    #     whiskerprops=dict(color="black"),
    #     capprops=dict(color="black"),
    #     medianprops=dict(color="black"),
    #     ax=ax1
    # )
    # # plot major points in blue, minor in orange
    # sns.swarmplot(
    #     x="type", y="prob_5ss",
    #     data=df_extremes[df_extremes["type"] == "major"],
    #     order=["minor", "major"],
    #     color="blue", size=12, ax=ax1
    # )
    # sns.swarmplot(
    #     x="type", y="prob_5ss",
    #     data=df_extremes[df_extremes["type"] == "minor"],
    #     order=["minor", "major"],
    #     color="orange", size=12, ax=ax1
    # )
    # ax1.set_ylabel("5'SS Score", fontsize=40)
    # ax1.set_xlabel("")
    # ax1.set_xticks([0, 1])
    # ax1.set_xticklabels(
    #     ["Minor", "Major"],
    #     fontsize=40
    # )

    # ax1.xaxis.set_tick_params(labelsize=40)
    # ax1.yaxis.set_tick_params(labelsize=40)
    # # ax1.set_title("Score Distribution", fontsize=13)
    # ax1.set_xlabel("Alternative 5'SS", fontsize=40)
    # ax1.set_yticks([0, 1])

    # --- Left: transparent boxplot + colored swarm + connecting lines ---
    sns.boxplot(
        x="type", y="prob_5ss", data=df_extremes,
        order=["minor", "major"],
        showfliers=False,
        boxprops=dict(facecolor="none", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
        ax=ax1
    )
    # 先畫連線
    for idx, grp in df_extremes.groupby("index"):
        if grp["type"].nunique() == 2:
            y_minor = grp.loc[grp["type"] == "minor", "prob_5ss"].values[0]
            y_major = grp.loc[grp["type"] == "major", "prob_5ss"].values[0]
            ax1.plot([0, 1], [y_minor, y_major],
                     color="gray", linewidth=1.2, alpha=1, zorder=0)
    # 再畫 swarmplot
    sns.swarmplot(
        x="type", y="prob_5ss",
        data=df_extremes,
        order=["minor", "major"],
        hue="type", palette={"minor":"orange","major":"blue"},
        size=12, ax=ax1, dodge=False
    )

    ax1.set_ylabel("5'SS Score", fontsize=40)
    ax1.set_xlabel("Alternative 5'SS", fontsize=40)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Minor", "Major"], fontsize=40)
    ax1.xaxis.set_tick_params(labelsize=40)
    ax1.yaxis.set_tick_params(labelsize=40)
    ax1.set_yticks([0, 1])


    # --- Right: scatter + Pearson r ---
    jitter = 0.5
    x = df_extremes["PSI"] + np.random.normal(scale=jitter, size=len(df_extremes))
    y = df_extremes["prob_5ss"]
    colors = df_extremes["type"].map({"major":"blue","minor":"orange"})
    ax2.scatter(x, y, c=colors, s=150, alpha=1.0, edgecolors='none')
    ax2.set_xlabel("PSI", fontsize=40)
    ax2.set_ylabel("5'SS Score", fontsize=40)
    # ax2.set_title(f"Pearson r = {r:.2f}, p = {p:.2g}", fontsize=13)

    # 新增一個自訂圖例項目來顯示 r 和 p 值
    # legend_text = f"Pearson r = {r:.2f}**"
    legend_text = f"r = {r:.2f}**"
    # legend_text = f"r = {r:.2f}**\n(p = {p:.2g})"
    ax2.set_xticks([0, 100])
    ax2.set_yticks([0, 1])

    
    custom_legend = [Line2D([0], [0], color='none', label=legend_text)]

    # 設定 legend
    # ax2.legend(handles=custom_legend, loc='best', fontsize=40, frameon=False)
    ax2.legend(
        handles=custom_legend,
        loc='upper left',  # 基準點是左上角
        bbox_to_anchor=(-0.1, 1.07),  # 往上移一點（y > 1）
        fontsize=40,
        frameon=False
    )

    ax2.xaxis.set_tick_params(labelsize=40)
    ax2.yaxis.set_tick_params(labelsize=40)

    # ax2.legend(
    #     handles=[
    #         plt.Line2D([0],[0], marker='o', color='w', label="Major", markerfacecolor='blue',  markersize=8),
    #         plt.Line2D([0],[0], marker='o', color='w', label="Minor", markerfacecolor='orange', markersize=8)
    #     ],
    #     title="Event", loc="lower left", fontsize=40
    # )

    # ax2.legend(
    #     handles=[
    #         plt.Line2D([0],[0], marker='o', color='w', label="Major", markerfacecolor='blue',  markersize=8),
    #         plt.Line2D([0],[0], marker='o', color='w', label="Minor", markerfacecolor='orange', markersize=8)
    #     ],
    #     title="Event", loc="lower left", fontsize=30
    # )


    plt.tight_layout()
    plt.savefig("./A5_result/A5_t_all.png", dpi=300)
    plt.show()
    print(f"r = {r:.2f}**\n(p = {p:.2g})")

if __name__ == "__main__":
    main()
