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

checkpoint_interval = 40
print("checkpoint_interval = ",  checkpoint_interval)

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



with open(f't_predictions_new_{my_seed}.pkl', 'rb') as f:
    data = pickle.load(f)

# 保存原始的 sys.stdout
original_stdout = sys.stdout
 
# 1117
for i in [603, 912]:  # 循环遍历指定索引范围
   
    print(f"Running index {i}... len = {lengths[i]}")

    # continue

    # if lengths[i] <= 5000 or lengths[i] > 10000:
    #     print(f"Skipping index {i} because the sequence is too long ({lengths[i]}).")
    #     continue
    
    fold_name = f"./{my_seed}_t_intron_score_all_parse/t_result_RI"
    output_filename = f"./{my_seed}_t_intron_score_all_parse/t_result_RI/000_arabidopsis_g_{i}.txt"  # 注意这里用 i 作为文件名的一部分，避免重复写入同一个文件
    import os
    os.makedirs(fold_name, exist_ok=True)


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


    intron_dict, logZ = SMsplice.forward_backward_low_memory_intron(
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

    print("Partition function (logZ) =", logZ)
    print("#Introns =", len(intron_dict))

    # set_trace()

    all_intron =  list(intron_dict.items())
    sorted_intron_dict = sorted(all_intron, key=lambda x: x[1], reverse=True)

    print("5SS, 3SS, prob")
    for (a_pos, b_pos), prob in sorted_intron_dict:
        print(f"{a_pos}, {b_pos-1}, {prob}")

  


    sys.stdout.close()
    sys.stdout = original_stdout

