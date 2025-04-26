import numpy as np
import pandas as pd
import time, argparse
from Bio import SeqIO, SeqUtils, motifs
from Bio.Seq import Seq 
from Bio.SeqRecord import SeqRecord
import scipy.ndimage
import scipy.stats as stats
import pickle
import SMsplice
from ipdb import set_trace
import sys

my_seed = int(sys.argv[1])  # 读取命令行参数并转换为整数
# top_k = int(sys.argv[2])

# top_k = np.inf

# my_seed = 2
np.random.seed(my_seed)
print(f"seed = {my_seed}")
print(f"all parse")

# if len(sys.argv) < 3:
#     print("Usage: python runSMsplice_fba.py <index> <top_k>")
#     sys.exit(1)


checkpoint_interval = 100

# 加载 pickle 文件中的字典
with open(f't_new_{my_seed}.pkl', 'rb') as f:
    data = pickle.load(f)

# 提取各个变量
sequences = data['sequences']
pME = data['pME']
pELF = data['pELF']
pIL = data['pIL']
pEE = data['pEE']
pELM = data['pELM']
pEO = data['pEO']
pELL = data['pELL']
emissions5 = data['emissions5']
emissions3 = data['emissions3']
lengths = data['lengths']
trueSeqs = data['trueSeqs']
testGenes = data['testGenes']
B3 = data['B3']
B5 = data['B5']

# set_trace()

# with open(f't_pred_all_new_{my_seed}.pkl', 'rb') as f:
#     data_pred_all = pickle.load(f)

# pred_all = data_pred_all['pred_all']
# loglik = pred_all[1]

with open(f't_predictions_new_{my_seed}.pkl', 'rb') as f:
    data = pickle.load(f)

# 保存原始的 sys.stdout
original_stdout = sys.stdout
 
# [23, 34, 51, 102, 112, 122, 150, 187, 223, 258, 336, 343, 390, 399, 415, 439, 457, 483, 591, 599, 603, 629, 650, 663, 664, 686, 705, 767, 781, 836, 857, 866, 889, 890, 903, 907, 912, 990, 1021, 1052]

# 23, 34, 889, 781, 767, 599, 415, 399, 258

for i in [258]:  # 循环遍历指定索引范围
    # 打印当前正在处理的索引，输出到原始标准输出（终端）
    print(f"Running index {i}... len = {lengths[i]}")

    # if lengths[i] <= 5000 or lengths[i] > 10000:
    #     print(f"Skipping index {i} because the sequence is too long ({lengths[i]}).")
    #     continue

    original_stdout.write(f"Running index {i}...\n")
    
    fold_name = f"./{my_seed}_t_exon_score_all_parse/t_result_SE"
    output_filename = f"./{my_seed}_t_exon_score_all_parse/t_result_SE/000_arabidopsis_g_{i}.txt"  # 注意这里用 i 作为文件名的一部分，避免重复写入同一个文件
    import os
    os.makedirs(fold_name, exist_ok=True)

    # 重定向输出到文件
    sys.stdout = open(output_filename, "w")
    
    print(f"Gene = {testGenes[i]}")
    print(f"index = {i}")

    trueFives_all = data['trueFives_all']
    trueThrees_all = data['trueThrees_all']

    print(f"Annotated 5SS: {trueFives_all[i]}")
    print(f"Annotated 3SS: {trueThrees_all[i]}")

    predFives_all = data['predFives_all']
    predThrees_all = data['predThrees_all']

    print(f"SMsplice 5SS: {predFives_all[i]}")
    print(f"SMsplice 3SS: {predThrees_all[i]}")

    print(f"length: {lengths[i]}")


    first_exon_dict, middle_exon_dict, logZ, F_current, B_current = SMsplice.forward_backward_low_memory_exon(
        sequences[i],
        pME,
        pELF,
        pIL,
        pEE,
        pELM,
        pEO,
        pELL,
        emissions5[i],
        emissions3[i],
        lengths[i],
        checkpoint_interval,
        0
    )


    last_exon_dict, common_logZ = SMsplice.forward_backward_last_exon(
        F_current,
        B_current,
        lengths[i],
        logZ
    )

  

    all_exon = list(first_exon_dict.items()) \
            + list(middle_exon_dict.items())       \
            + list(last_exon_dict.items())

    print("Partition function (logZ) =", logZ)
    print("#Exons =", len(all_exon))



    sorted_all_exon = sorted(all_exon, key=lambda x: x[1], reverse=True)

    print("3SS, 5SS, prob")
    for (a_pos, b_pos), prob in sorted_all_exon:
        print(f"{a_pos}, {b_pos-1}, {prob}")


    sys.stdout.close()
    sys.stdout = original_stdout