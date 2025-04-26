
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute intron-score statistics and make plots.
Key point: the *merged* plot now shows **exactly two boxes per species**
(TP vs. FP) by adding `hue='Pred'` and `dodge=True`
to the box-plot call—and we’ve increased `width` to pull them apart.
"""

import os, re, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# switches / style
# ---------------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_context("notebook")
os.makedirs("0_intron_png", exist_ok=True)

# ---------------------------------------------------------------------
# helper: parse splice site block
# ---------------------------------------------------------------------
def parse_splice_file(fname: str) -> pd.DataFrame:
    txt = open(fname).read()

    def _grab_set(pat):
        m = re.search(pat, txt)
        if not m or not m.group(1).strip():
            return set()
        return set(map(int, re.split(r"[\s,]+", m.group(1).strip())))

    ann5 = _grab_set(r"Annotated 5SS:\s*\[([^\]]*)\]")
    ann3 = _grab_set(r"Annotated 3SS:\s*\[([^\]]*)\]")
    sm5  = _grab_set(r"SMsplice 5SS:\s*\[([^\]]*)\]")
    sm3  = _grab_set(r"SMsplice 3SS:\s*\[([^\]]*)\]")

    # blk5 = re.compile(r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′])", re.S)
    # blk3 = re.compile(r"Sorted 3['′] Splice Sites .*?\n(.*)", re.S)
    line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")

    rows = []
    # if blk5.search(txt):
    #     for a, b in line.findall(blk5.search(txt).group(1)):
    #         rows.append((int(a), float(b), "5prime", int(a) in ann5, int(a) in sm5))
    # if blk3.search(txt):
    #     for a, b in line.findall(blk3.search(txt).group(1)):
    #         rows.append((int(a), float(b), "3prime", int(a) in ann3, int(a) in sm3))

    e5 = {r[0] for r in rows if r[2] == "5prime"}
    e3 = {r[0] for r in rows if r[2] == "3prime"}
    for p in ann5 - e5: rows.append((p, 0, "5prime", True,  p in sm5))
    for p in ann3 - e3: rows.append((p, 0, "3prime", True,  p in sm3))
    for p in sm5  - e5: rows.append((p, 0, "5prime", False, True))
    for p in sm3  - e3: rows.append((p, 0, "3prime", False, True))

    return pd.DataFrame(rows, columns=["pos","prob","type","is_TP","is_viterbi"])

def prob5(p, df):
    sub = df[(df.type=="5prime") & df.is_viterbi & (df.pos==p)]
    return sub.prob.max() if not sub.empty else 0

def prob3(p, df):
    sub = df[(df.type=="3prime") & df.is_viterbi & (df.pos==p)]
    return sub.prob.max() if not sub.empty else 0

# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
species_map = {
    'h':'Human',
    'm':'Mouse',
    'z':'Zebrafish',
    't':'Arabidopsis',
    'o':'Moth',
    'f':'Fly'
}
seeds, top_ks = [0], [1000]

all_intron_scores = []
# all_scores = []

# ---------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------
for seed in tqdm(seeds, desc="seed"):
    for top_k in tqdm(top_ks, desc="top_k", leave=False):
        for code, species in species_map.items():
            pattern = f"./{seed}_{code}_intron_score/{code}_result_{top_k}/000_{species.lower()}_g_*.txt"
            files = glob.glob(pattern)
            if not files:
                continue

            corr, inc = [], []
            for fname in files:
                txt = open(fname).read()
                ann5 = list(map(int, re.findall(r'\d+', re.search(r"Annotated 5SS:\s*\[([^\]]*)\]", txt).group(1))))
                ann3 = list(map(int, re.findall(r'\d+', re.search(r"Annotated 3SS:\s*\[([^\]]*)\]", txt).group(1))))
                sm5  = list(map(int, re.findall(r'\d+', re.search(r"SMsplice 5SS:\s*\[([^\]]*)\]", txt).group(1))))
                sm3  = list(map(int, re.findall(r'\d+', re.search(r"SMsplice 3SS:\s*\[([^\]]*)\]", txt).group(1))))
                df_raw = parse_splice_file(fname)

                # optional table
                intron_table = {}
                for line in txt.splitlines():
                    if line and line[0].isdigit():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                intron_table[(int(parts[0]), int(parts[1]))] = float(parts[2])
                            except:
                                pass

                # build pairs for intron predictions
                if not ann5 or not sm5 or not ann3 or not sm3:
                    continue
                ann_pairs = (
                    list(zip(sorted(ann5), sorted(ann3))) 
                )
                sm_pairs = (
                    list(zip(sorted(sm5), sorted(sm3))) 
                )


                for a, b in sm_pairs:
                    # p3 = df_raw[(df_raw.type=="3prime") & df_raw.is_viterbi & (df_raw.pos==a)].prob.max() if a!=0 else 1.0
                    # p5 = df_raw[(df_raw.type=="5prime") & df_raw.is_viterbi & (df_raw.pos==b)].prob.max() if b!=-1 else 1.0
                    score = intron_table.get((a,b), 0)
                    if score == 0:
                        continue
                    ok = True if (a, b) in ann_pairs else False
                    label = 'TP' if ok else 'FP'
                    all_intron_scores.append({
                        'species': code,
                        'score':   score,
                        'Pred':    label
                    })

                # for a5, a3 in zip(sm5, sm3):
                #     score = table.get((a5, a3), prob5(a5, df_raw) * prob3(a3, df_raw))
                #     if score == 0:
                #         continue
                #     ok = (a5 in ann5) and (a3 in ann3)
                #     all_scores.append({
                #         'species': code,
                #         'score': score,
                #         'label': 'TP' if ok else 'FP'
                #     })
                #     (corr if ok else inc).append(score)

            # total = len(corr) + len(inc)
            # high = [s for s in corr+inc if s >= 0.9]
            # results.append({
            #     'seed': seed,
            #     'top_k': top_k,
            #     'species': code,
            #     'precision_09': len([s for s in corr if s>=0.9]) / len(high) if high else None,
            #     'subset_09'  : len(high) / total if total else None
            # })

df = pd.DataFrame(all_intron_scores)
# map to title‐case species names
df['species'] = df['species'].map(species_map)

plt.figure(figsize=(12.5, 6))
ax = sns.boxplot(
    x='species', y='score', hue='Pred', data=df,
    hue_order=['TP','FP'],
    dodge=True, width=0.8,
    palette={'TP':'#1f77b4','FP':'#ff7f0e'},
    showcaps=True, showfliers=False,
    # boxprops={'facecolor':'white','edgecolor':'black'},
    # whiskerprops={'color':'black'}, capprops={'color':'black'},
    # medianprops={'color':'black'}
    # 線寬全部調大
    boxprops     = {'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 2},
    whiskerprops = {'color': 'black', 'linewidth': 2},
    capprops     = {'color': 'black', 'linewidth': 2},
    medianprops  = {'color': 'black', 'linewidth': 2}
)

sns.stripplot(
    x='species', y='score', hue='Pred', data=df,
    hue_order=['TP','FP'],
    dodge=True, jitter=0.2, size=1.5, linewidth=0,
    palette={'TP':'#1f77b4','FP':'#ff7f0e'},
    ax=ax
)

# formatting
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
point_handles = [
    Line2D([0],[0], marker='o', color='#1f77b4', linestyle='', markersize=10),
    Line2D([0],[0], marker='o', color='#ff7f0e', linestyle='', markersize=10)
]
ax.legend(
    point_handles, ['TP','FP'],
    fontsize=25, loc='upper center',
    bbox_to_anchor=(0.5,1.15), ncol=2, frameon=False
)
ax.set_ylabel('Intron Score', fontsize=25)
ax.tick_params(labelsize=25)
sns.despine()
plt.tight_layout()
plt.savefig("merged_intron_box_strip.png", dpi=300)
plt.close()

print("Merged intron plot saved → merged_intron_box_strip.png")


