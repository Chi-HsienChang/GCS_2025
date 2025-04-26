#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute exon-score statistics, calculate precision at score ≥ 0.9, and
plot a merged box + strip plot across species. Two boxes per species
(TP vs. FP) with narrow widths for extra gap and a custom point legend
on top with rotated x-labels.
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
trace = True
sns.set_style("whitegrid")
sns.set_context("notebook")
os.makedirs("0_exon_png", exist_ok=True)

# ---------------------------------------------------------------------
# helper: parse splice site block from a file
# ---------------------------------------------------------------------
def parse_splice_file(filename):
    with open(filename, "r") as f:
        text = f.read()
    def parse_list(regex):
        m = regex.search(text)
        if not m or not m.group(1).strip():
            return set()
        return set(map(int, re.split(r"[\s,]+", m.group(1).strip())))
    pat5 = re.compile(r"Annotated 5SS:\s*\[([^\]]*)\]")
    pat3 = re.compile(r"Annotated 3SS:\s*\[([^\]]*)\]")
    p5v  = re.compile(r"SMsplice 5SS:\s*\[([^\]]*)\]")
    p3v  = re.compile(r"SMsplice 3SS:\s*\[([^\]]*)\]")
    ann5 = parse_list(pat5)
    ann3 = parse_list(pat3)
    v5   = parse_list(p5v)
    v3   = parse_list(p3v)
    if trace:
        print("annotated 5′SS:", ann5)
        print("annotated 3′SS:", ann3)
        print("viterbi 5′SS:", v5)
        print("viterbi 3′SS:", v3)
    blk5 = re.compile(r"Sorted 5['′] Splice Sites .*?\n(.*?)\n(?=Sorted 3['′] Splice Sites)", re.S)
    blk3 = re.compile(r"Sorted 3['′] Splice Sites .*?\n(.*)", re.S)
    line = re.compile(r"Position\s+(\d+)\s*:\s*([\d.eE+-]+)")
    rows = []
    if blk5.search(text):
        for a, b in line.findall(blk5.search(text).group(1)):
            rows.append((int(a), float(b), "5prime", int(a) in ann5, int(a) in v5))
    if blk3.search(text):
        for a, b in line.findall(blk3.search(text).group(1)):
            rows.append((int(a), float(b), "3prime", int(a) in ann3, int(a) in v3))
    e5 = {r[0] for r in rows if r[2]=="5prime"}
    e3 = {r[0] for r in rows if r[2]=="3prime"}
    for p in ann5 - e5: rows.append((p,0,"5prime",True,  p in v5))
    for p in ann3 - e3: rows.append((p,0,"3prime",True,  p in v3))
    for p in v5  - e5: rows.append((p,0,"5prime",False, True))
    for p in v3  - e3: rows.append((p,0,"3prime",False, True))
    return pd.DataFrame(rows, columns=["pos","prob","type","is_TP","is_viterbi"])

# ---------------------------------------------------------------------
# config: species list and seeds/top_k values
# ---------------------------------------------------------------------
species_map = {
    'h':'Human',
    'm':'Mouse',
    'z':'Zebrafish',
    't':'Arabidopsis',
    'o':'Moth',
    'f':'Fly'
}
seeds = [0]
top_ks = [1000]

all_exon_scores = []

# ---------------------------------------------------------------------
# main loop: parse files, compute exon scores (TP vs FP)
# ---------------------------------------------------------------------
for seed in tqdm(seeds, desc="Seeds"):
    for top_k in tqdm(top_ks, desc="top_k", leave=False):
        for code, species in species_map.items():
            pattern = f"./{seed}_{code}_exon_score/{code}_result_{top_k}/000_{species.lower()}_g_*.txt"
            files = glob.glob(pattern)
            if not files:
                continue
            for fname in files:
                content = open(fname).read()
                ann5 = set(map(int, re.findall(r"Annotated 5SS:\s*\[([^\]]*)\]", content)[0].split()))
                ann3 = set(map(int, re.findall(r"Annotated 3SS:\s*\[([^\]]*)\]", content)[0].split()))
                sm5  = set(map(int, re.findall(r"SMsplice 5SS:\s*\[([^\]]*)\]", content)[0].split()))
                sm3  = set(map(int, re.findall(r"SMsplice 3SS:\s*\[([^\]]*)\]", content)[0].split()))
                df_raw = parse_splice_file(fname)
                exon_table = {}
                for line in content.splitlines():
                    if line and line[0].isdigit():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                exon_table[(int(parts[0]), int(parts[1]))] = float(parts[2])
                            except:
                                pass
                ann_pairs = set(
                    [(0, min(ann5))] +
                    list(zip(sorted(ann3), sorted(ann5)[1:])) +
                    [(max(ann3), -1)]
                )
                sm_pairs = (
                    [(0, min(sm5))] +
                    list(zip(sorted(sm3), sorted(sm5)[1:])) +
                    [(max(sm3), -1)]
                )
                for a, b in sm_pairs:
                    score = exon_table.get((a,b), 0.0)
                    if score == 0.0:
                        continue
                    ok = (a,b) in ann_pairs
                    all_exon_scores.append({
                        'species': code,
                        'score':   score,
                        'Pred':    'TP' if ok else 'FP'
                    })

# ---------------------------------------------------------------------
# compute precision@0.9 and save summary
# ---------------------------------------------------------------------
df = pd.DataFrame(all_exon_scores)
df['Species'] = df['species'].map(species_map)

summary = []
for code, name in species_map.items():
    sub = df[df['species']==code]
    high = sub[sub['score'] >= 0.9]
    prec = high['Pred'].eq('TP').sum() / len(high) if len(high)>0 else np.nan
    frac = len(high) / len(sub) if len(sub)>0 else np.nan
    summary.append({'Species': name, 'precision_09': prec, 'fraction_09': frac})

pd.DataFrame(summary).to_csv("exon_precision_0.9.csv", index=False)
print("Saved exon precision@0.9 → exon_precision_0.9.csv")

# ---------------------------------------------------------------------
# plotting: merged box + strip, narrow boxes, custom legend, rotated ticks
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
    palette={'TP':'#1f77b4','FP':'#ff7f0e'},
    ax=ax
)

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

ax.set_ylabel('Exon Score', fontsize=25)
ax.tick_params(labelsize=25)
sns.despine()
plt.tight_layout()
plt.savefig("merged_exon_box_strip.png", dpi=300)
plt.close()

print("Merged exon plot saved → merged_exon_box_strip.png")
