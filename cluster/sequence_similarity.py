import numpy as np

"""
please use python 2
"""


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


protein_vocabulary_dict = {}
f = open("./smith-waterman-src/protein.fasta")  # peptide.fasta
i = 0
for line in f.readlines():
    if line[0] == ">":
        protein_vocabulary_dict[line[1:-1]] = i
        i += 1
f.close()
print
len(protein_vocabulary_dict)
# print protein_vocabulary_dict.keys()
p_simi = np.zeros((len(protein_vocabulary_dict), len(protein_vocabulary_dict)))

count = 0
f = open("./smith-waterman-src/protein_output.txt")  # peptide_output.txt
lines = f.readlines()
f.close()
print("total lines", len(lines))

for i in range(0, len(lines), 4):
    try:
        a = lines[i].strip("\n").split(" ")[-1]
        b = lines[i + 1].strip("\n").split(" ")[-1]
        c = float(int(lines[i + 2].strip("\n").split()[1]))
        p_simi[protein_vocabulary_dict[a], protein_vocabulary_dict[b]] = c
    except:
        print(
            "wrong", i, a, b, c, protein_vocabulary_dict[a], protein_vocabulary_dict[b]
        )
        exit()

print(check_symmetric(p_simi))

for i in xrange(len(p_simi)):
    for j in xrange(len(p_simi)):
        if i == j:
            continue
        p_simi[i, j] = p_simi[i, j] / (
            float(np.sqrt(p_simi[i, i]) * np.sqrt(p_simi[j, j])) + 1e-12
        )
for i in xrange(len(p_simi)):
    p_simi[i, i] = p_simi[i, i] / float(np.sqrt(p_simi[i, i]) * np.sqrt(p_simi[i, i]))

print
"p_simi", p_simi.shape
print(check_symmetric(p_simi))
np.save("seq_sim_mat", p_simi)
