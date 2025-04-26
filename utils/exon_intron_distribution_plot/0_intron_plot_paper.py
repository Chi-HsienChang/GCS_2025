#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute intron-score statistics, calculate precision at score ≥ 0.9, and
make plots. The merged plot shows two boxes per species (TP vs. FP) with
narrow box widths for extra gap and a custom point legend on top with
rotated x-labels.
"""

import os
import re
import glob
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

    # We only need to fill missing entries with prob=0
    rows = []
    e5 = set()
    e3 = set()
    # add annotated-only and viterbi-only entries
    for p in ann5:
        rows.append((p, 0.0, "intron", True,  p in sm5)); e5.add(p)
    for p in ann3:
        rows.append((p, 0.0, "intron", True,  p in sm3)); e3.add(p)
    for p in sm5 - e5:
        rows.append((p, 0.0, "intron", False, True))
    for p in sm3 - e3:
        rows.append((p, 0.0, "intron", False, True))

    return pd.DataFrame(rows, columns=["start","prob","type","is_TP","is_viterbi"])

# ---------------------------------------------------------------------
# config
# ---------------------------------------------------------------------
species_map = {
    'h': 'Human',
    'm': 'Mouse',
    'z': 'Zebrafish',
    't': 'Arabidopsis',
    'o': 'Moth',
    'f': 'Fly'
}
seeds, top_ks = [0], [1000]

all_scores = []

# ---------------------------------------------------------------------
# main loop: collect TP/FP scores
# ---------------------------------------------------------------------
for seed in tqdm(seeds, desc="seed"):
    for top_k in tqdm(top_ks, desc="top_k", leave=False):
        for code, species in species_map.items():
            pattern = f"./{seed}_{code}_intron_score/{code}_result_{top_k}/000_{species.lower()}_g_*.txt"
            files = glob.glob(pattern)
            if not files:
                continue

            # build annotated pairs set
            for fname in files:
                txt = open(fname).read()
                ann5 = set(map(int, re.findall(r"Annotated 5SS:\s*\[([^\]]*)\]", txt)[0].split()))
                ann3 = set(map(int, re.findall(r"Annotated 3SS:\s*\[([^\]]*)\]", txt)[0].split()))
                sm5  = set(map(int, re.findall(r"SMsplice 5SS:\s*\[([^\]]*)\]", txt)[0].split()))
                sm3  = set(map(int, re.findall(r"SMsplice 3SS:\s*\[([^\]]*)\]", txt)[0].split()))

                # optional table of intron scores
                table = {}
                for line in txt.splitlines():
                    if line and line[0].isdigit():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                table[(int(parts[0]), int(parts[1]))] = float(parts[2])
                            except:
                                pass

                # true positive pairs
                ann_pairs = set(zip(sorted(ann5), sorted(ann3)))
                # predicted pairs
                sm_pairs  = list(zip(sorted(sm5), sorted(sm3)))

                for a, b in sm_pairs:
                    score = table.get((a,b), 0.0)
                    if score == 0.0:
                        continue
                    ok = (a,b) in ann_pairs
                    all_scores.append({
                        'species': code,
                        'score':   score,
                        'Pred':    'TP' if ok else 'FP'
                    })

# ---------------------------------------------------------------------
# build DataFrame and compute precision@0.9
# ---------------------------------------------------------------------
df = pd.DataFrame(all_scores)
df['Species'] = df['species'].map(species_map)

summary = []
for code, name in species_map.items():
    sub = df[df['species']==code]
    high = sub[sub['score']>=0.9]
    prec = high['Pred'].eq('TP').sum() / len(high) if len(high)>0 else np.nan
    subset = len(high) / len(sub) if len(sub)>0 else np.nan
    summary.append({'Species': name, 'precision_09': prec, 'subset_09': subset})
pd.DataFrame(summary).to_csv("intron_precision_0.9.csv", index=False)

# ---------------------------------------------------------------------
# plotting: merged box + stripplot
# ---------------------------------------------------------------------
plt.figure(figsize=(12.5, 6))
ax = sns.boxplot(
    x='Species', y='score', hue='Pred', data=df,
    hue_order=['TP','FP'],
    dodge=True, width=0.4,
    palette={'TP':'#1f77b4','FP':'#ff7f0e'},
    showcaps=True, showfliers=False,
    boxprops={'facecolor':'white','edgecolor':'black'},
    whiskerprops={'color':'black'}, capprops={'color':'black'},
    medianprops={'color':'black'}
)
sns.stripplot(
    x='Species', y='score', hue='Pred', data=df,
    hue_order=['TP','FP'],
    dodge=True, jitter=0.2, size=1.5, linewidth=0,
    palette={'TP':'#1f77b4','FP':'#ff7f0e'}, ax=ax
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

print("Intron precision@0.9 saved → intron_precision_0.9.csv")
print("Merged intron plot saved → merged_intron_box_strip.png")
