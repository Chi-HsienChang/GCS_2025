#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4×1 summary figure (shared y-axis), x-axis 显示 species（Arabidopsis(all), Arabidopsis(top-k), Human(top-k), Mouse(top-k)）
但不再有“dataset”这一行总标签

Rows: SE, RI, A5SS, A3SS
Hue: classification (blue = first category, orange = second)
Boxes transparent, swarm points centered within each box, Alt5/Alt3 pair‐lines drawn.

USER SETTINGS:
  point_size_map = { (event, dataset): size, ... }
  legend_y_offset = how far above each subplot to place its legend
"""
import os, re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ─── USER SETTINGS ────────────────────────────────────────────────────────────
point_size_map = {
    ("SE", "Arabidopsis (all)"):   5,
    ("SE", "Arabidopsis (top-k)"): 5,
    ("SE", "Human (top-k)"):       5,
    ("SE", "Mouse (top-k)"):       3.5,
    ("RI", "Arabidopsis (all)"):   5,
    ("RI", "Arabidopsis (top-k)"): 5,
    ("RI", "Human (top-k)"):       3,
    ("RI", "Mouse (top-k)"):       1.5,
    ("A5", "Arabidopsis (all)"):   5,
    ("A5", "Arabidopsis (top-k)"): 5,
    ("A5", "Human (top-k)"):       5,
    ("A5", "Mouse (top-k)"):       5,
    ("A3", "Arabidopsis (all)"):   5,
    ("A3", "Arabidopsis (top-k)"): 5,
    ("A3", "Human (top-k)"):       5,
    ("A3", "Mouse (top-k)"):       5,
}
legend_y_offset = 1.31 # move legends further up

# ─── Constants ───────────────────────────────────────────────────────────────
DATASETS = [
    "Arabidopsis (all)",
    "Arabidopsis (top-k)",
    "Human (top-k)",
    "Mouse (top-k)",
]
X_LABELS = [
    "Arabidopsis\n(all)",
    "Arabidopsis\n(top-k)",
    "Human\n(top-k)",
    "Mouse\n(top-k)",
]
EVENTS = ["SE", "RI", "A5", "A3"]
TITLE_MAP = {
    "SE": "SE",
    "RI": "RI",
    "A5": "A5SS",
    "A3": "A3SS",
}
ORDERS = {
    "SE": ["non-SE", "SE"],
    "RI": ["non-RI", "RI"],
    "A5": ["Major",   "Minor"],
    "A3": ["Major",   "Minor"],
}
PALETTE = ["blue", "orange"]
YLABELS = {
    "SE": "Exon Score",
    "RI": "Intron Score",
    "A5": "5'SS Score",
    "A3": "3'SS Score",
}

PATHS = {
    ("SE", "Arabidopsis (all)"):   ("./data/tair10_SE.txt", "./t_allParse_exon"),
    ("SE", "Arabidopsis (top-k)"): ("./data/tair10_SE.txt", "./t_result_exon_1000"),
    ("SE", "Human (top-k)"):       ("./data/hg19_SE.txt",   "./Human_Mouse/0_h_exon_score/h_result_1000"),
    ("SE", "Mouse (top-k)"):       ("./data/mm10_SE.txt",   "./Human_Mouse/0_m_exon_score/m_result_1000"),
    ("RI", "Arabidopsis (all)"):   ("./data/tair10_RI.txt","./t_allParse_intron"),
    ("RI", "Arabidopsis (top-k)"): ("./data/tair10_RI.txt","./t_result_intron_1000"),
    ("RI", "Human (top-k)"):       ("./data/hg19_RI.txt",  "./Human_Mouse/0_h_intron_score/h_result_1000"),
    ("RI", "Mouse (top-k)"):       ("./data/mm10_RI.txt",  "./Human_Mouse/0_m_intron_score/m_result_1000"),
    ("A5", "Arabidopsis (all)"):   ("./data/tair10_A5.txt","./t_allParse_ss"),
    ("A5", "Arabidopsis (top-k)"): ("./data/tair10_A5.txt","./t_result_ss_1000"),
    ("A5", "Human (top-k)"):       ("./data/hg19_A5.txt",  "./Human_Mouse/0_h_ss_score/h_result_1000"),
    ("A5", "Mouse (top-k)"):       ("./data/mm10_A5.txt",  "./Human_Mouse/0_m_ss_score/m_result_1000"),
    ("A3", "Arabidopsis (all)"):   ("./data/tair10_A3.txt","./t_allParse_ss"),
    ("A3", "Arabidopsis (top-k)"): ("./data/tair10_A3.txt","./t_result_ss_1000"),
    ("A3", "Human (top-k)"):       ("./data/hg19_A3.txt",  "./Human_Mouse/0_h_ss_score/h_result_1000"),
    ("A3", "Mouse (top-k)"):       ("./data/mm10_A3.txt",  "./Human_Mouse/0_m_ss_score/m_result_1000"),
}

SEC5_RE = re.compile(
    r"Sorted 5' Splice Sites \(High to Low Probability\):([\s\S]+?)Sorted 3' Splice Sites",
    re.MULTILINE,
)
SEC3_RE = re.compile(
    r"Sorted 3' Splice Sites \(High to Low Probability\):([\s\S]+)",
    re.MULTILINE,
)
POS_RE  = re.compile(r"Position\s*(\d+):\s*([0-9.eE+\-]+)")
LINE_RE = re.compile(r"^\s*(\d+),\s*(\d+),\s*([0-9.eE+\-]+)\s*$")

# ─── Data loaders ────────────────────────────────────────────────────────────
def load_se_ri(event, csv_path, txt_folder):
    df = pd.read_csv(csv_path, sep="\t")
    df["idx"], df["prob"] = -1, -1.0

    gene_map = {}
    for fn in os.listdir(txt_folder):
        if fn.endswith(".txt"):
            raw = Path(txt_folder, fn).read_text()
            mg = re.search(r"Gene\s*=\s*(\S+)", raw)
            mi = re.search(r"_g_(\d+)\.txt", fn)
            if mg and mi:
                gene_map[mg.group(1)] = (int(mi.group(1)), raw)

    for i, row in df.iterrows():
        gene, a5, a3 = row["gene"], row["5ss"], row["3ss"]
        if gene not in gene_map:
            continue
        idx, raw = gene_map[gene]
        df.at[i, "idx"] = idx
        for L in raw.splitlines():
            m = LINE_RE.match(L)
            if not m:
                continue
            v1, v2, p = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if (event == "SE" and v1 == a3 and v2 == a5) or (event == "RI" and v1 == a5 and v2 == a3):
                df.at[i, "prob"] = p
                break

    valid_idx = set(df[df["prob"] >= 0]["idx"])
    recs = []
    ann5_re = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    ann3_re = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    for fn in os.listdir(txt_folder):
        if not fn.endswith(".txt"):
            continue
        raw = Path(txt_folder, fn).read_text()
        im = re.search(r"index\s*=\s*(\d+)", raw)
        idx = int(im.group(1)) if im else None

        ann5 = list(map(int, re.findall(r"\d+", ann5_re.search(raw).group(1)))) if ann5_re.search(raw) else []
        ann3 = list(map(int, re.findall(r"\d+", ann3_re.search(raw).group(1)))) if ann3_re.search(raw) else []
        pairs = ([(ann5[i+1], ann3[i]) for i in range(min(len(ann3), len(ann5)-1))]
                 if event == "SE" else list(zip(ann5, ann3)))
        for five, three in pairs:
            cls = event if ((df["5ss"] == five) & (df["3ss"] == three)).any() else f"non-{event}"
            pval = None
            for L in raw.splitlines():
                m = LINE_RE.match(L)
                if not m:
                    continue
                v1, v2, p = int(m.group(1)), int(m.group(2)), float(m.group(3))
                cond = (event == "SE" and v1 == three and v2 == five) or (event == "RI" and v1 == five and v2 == three)
                if cond:
                    pval = p
                    break
            recs.append({"classification": cls, "prob": pval, "idx": idx})

    out = pd.DataFrame(recs, columns=["classification", "prob", "idx"])
    return out[out["idx"].isin(valid_idx) & out["prob"].notnull()]


def load_ss(event, csv_path, txt_folder):
    alt_field, prob_field, sec_re = {
        "A5": ("5ss_alternative", "prob_5ss", SEC5_RE),
        "A3": ("3ss_alternative", "prob_3ss", SEC3_RE),
    }[event]
    df = pd.read_csv(csv_path, sep="\t")
    df["idx"], df[prob_field] = -1, -1.0

    gene_map = {}
    for fn in os.listdir(txt_folder):
        if fn.endswith(".txt"):
            raw = Path(txt_folder, fn).read_text()
            mg = re.search(r"Gene\s*=\s*(\S+)", raw)
            mi = re.search(r"_g_(\d+)\.txt", fn)
            if mg and mi:
                gene_map[mg.group(1)] = (int(mi.group(1)), raw)

    for i, row in df.iterrows():
        gene = row["gene"]
        if gene not in gene_map:
            continue
        idx, raw = gene_map[gene]
        df.at[i, "idx"] = idx

        sec = sec_re.search(raw)
        if not sec:
            continue
        block = sec.group(1).splitlines()
        for L in block:
            m = POS_RE.match(L.strip())
            if m and int(m.group(1)) == int(row[alt_field]):
                df.at[i, prob_field] = float(m.group(2))
                break

    valid = df[df[prob_field] >= 0].copy()
    recs = []
    for idx_val in sorted(valid["idx"].unique()):
        grp = valid[valid["idx"] == idx_val]
        maj = grp.loc[grp["PSI"].idxmax()]
        minr = grp.loc[grp["PSI"].idxmin()]
        recs.append({"classification": "Major", "prob": maj[prob_field], "idx": idx_val})
        recs.append({"classification": "Minor", "prob": minr[prob_field], "idx": idx_val})

    out = pd.DataFrame(recs, columns=["classification", "prob", "idx"])
    cnt = out["idx"].value_counts()
    good = cnt[cnt == 2].index
    return out[out["idx"].isin(good)]


def main():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    # load all data
    rows = []
    for evt in EVENTS:
        for ds in DATASETS:
            csvp, txtf = PATHS[(evt, ds)]
            df = load_se_ri(evt, csvp, txtf) if evt in ["SE", "RI"] else load_ss(evt, csvp, txtf)
            df["event"], df["dataset"] = evt, ds
            rows.append(df)
    data = pd.concat(rows, ignore_index=True)

    # set up figure
    fig, axes = plt.subplots(4, 1, sharey=True, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.5)

    for ax, evt in zip(axes, EVENTS):
        sub = data[data["event"] == evt]
        order_cls = ORDERS[evt]
        pal = PALETTE

        # swarm points
        for ds in DATASETS:
            df_ds = sub[sub["dataset"] == ds]
            sns.swarmplot(
                x="dataset", y="prob", hue="classification", data=df_ds,
                order=[ds], hue_order=order_cls, palette=pal,
                dodge=True, size=point_size_map[(evt, ds)],
                ax=ax, legend=False, zorder=1
            )

        # transparent boxplots
        # sns.boxplot(
        #     x="dataset", y="prob", hue="classification", data=sub,
        #     order=DATASETS, hue_order=order_cls, palette=pal, ax=ax,
        #     boxprops=dict(facecolor="none", edgecolor="black", zorder=2),
        #     whiskerprops=dict(color="black", zorder=2),
        #     capprops=dict(color="black", zorder=2),
        #     medianprops=dict(color="black", zorder=2),
        #     showcaps=True, showfliers=False, dodge=True, width=0.8
        # )

        sns.boxplot(
            x="dataset", y="prob", hue="classification", data=sub,
            order=DATASETS, hue_order=order_cls, palette=pal, ax=ax,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=2, zorder=2),
            whiskerprops=dict(color="black", linewidth=2, zorder=2),
            capprops=dict(color="black", linewidth=2, zorder=2),
            medianprops=dict(color="black", linewidth=2, zorder=2),
            showcaps=True, showfliers=False, dodge=True, width=0.8
        )
        # Alt5/A3 pair-lines
        if evt in ["A5", "A3"]:
            for ds in DATASETS:
                grp_ds = sub[sub["dataset"] == ds]
                for _, grp in grp_ds.groupby("idx"):
                    if set(grp["classification"]) == set(order_cls):
                        x0 = DATASETS.index(ds)
                        y0 = grp.loc[grp["classification"] == order_cls[0], "prob"].iloc[0]
                        y1 = grp.loc[grp["classification"] == order_cls[1], "prob"].iloc[0]
                        ax.plot([x0 - 0.2, x0 + 0.2], [y0, y1],
                                color="gray", linewidth=1, alpha=0.7, zorder=1.5)

        # y-axis label
        ax.set_ylabel(YLABELS[evt])

        # # species ticks only
        ax.set_xticks(range(len(DATASETS)))
        # ax.set_xticklabels(X_LABELS, ha="center", fontsize=20)

        # only bottom row shows species labels
        if evt == EVENTS[-1]:
            # ax.set_xticks(range(len(DATASETS)))
            ax.set_xticklabels(X_LABELS, ha="center", fontsize=20)
        else:
            ax.set_xticks([])


        ax.set_xlabel("")

    

        # bold, left‐aligned title
        ax.set_title(TITLE_MAP[evt], loc="left", fontweight="bold")

        # legend above each subplot
        handles = [
            Line2D([0], [0], marker="o", color="w", label=cls,
                   markerfacecolor=pal[i], markersize=8)
            for i, cls in enumerate(order_cls)
        ]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, legend_y_offset),
                  ncol=2, frameon=False, fontsize=20)

    # final layout tweaks
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = Path("figs/combined_4row_sharedy.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
