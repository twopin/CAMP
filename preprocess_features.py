
import numpy as np
import sys
import pickle
import math

amino_acid_set = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 } # consider non-standard residues

amino_acid_num = 25

ss_set = {"H": 1, "C": 2, "E": 3} # revise order, not necessary if training your own model
ss_number = 3

physicochemical_set={'A': 1, 'C': 3, 'B': 7, 'E': 5, 'D': 5, 'G': 2, 'F': 1, 
			'I': 1, 'H': 6, 'K': 6, 'M': 1, 'L': 1, 'O': 7, 'N': 4, 
			'Q': 4, 'P': 1, 'S': 4, 'R': 6, 'U': 7, 'T': 4, 'W': 2, 
			'V': 1, 'Y': 4, 'X': 7, 'Z': 7}

residue_list = list(amino_acid_set.keys())
ss_list = list(ss_set.keys())


new_key_list = []
for i in residue_list:
    for j in ss_list:
        str_1 = str(i)+str(j)
        new_key_list.append(str_1)

new_value_list = [x+1 for x in list(range(amino_acid_num*ss_number))]

seq_ss_dict = dict(zip(new_key_list,new_value_list))
seq_ss_number = amino_acid_num*ss_number #75



def label_sequence(line, pad_prot_len, res_ind):
	X = np.zeros(pad_prot_len)

	for i, res in enumerate(line[:pad_prot_len]):
		X[i] = res_ind[res]

	return X

def label_seq_ss(line, pad_prot_len, res_ind):
	line = line.strip().split(',')
	X = np.zeros(pad_prot_len)
	for i ,res in enumerate(line[:pad_prot_len]):
		X[i] = res_ind[res]
	return X


def sigmoid(x):
	return 1 / (1 + math.exp(-x))

sigmoid_array=np.vectorize(sigmoid)

def padding_sigmoid_pssm(x,N):
	x = sigmoid_array(x)
	padding_array = np.zeros([N,x.shape[1]])
	if x.shape[0]>=N: # sequence is longer than N
		padding_array[:N,:x.shape[1]] = x[:N,:]
	else:
		padding_array[:x.shape[0],:x.shape[1]] = x
	return padding_array

def padding_intrinsic_disorder(x,N):
	padding_array = np.zeros([N,x.shape[1]])
	if x.shape[0]>=N: # sequence is longer than N
		padding_array[:N,:x.shape[1]] = x[:N,:]
	else:
		padding_array[:x.shape[0],:x.shape[1]] = x
	return padding_array



if __name__ == '__main__':
    input_file = sys.argv[1]
	f = open(input_file)
	pep_set = set()
	seq_set = set()
	pep_ss_set = set()
	seq_ss_set = set()

	for line in f.readlines()[1:]: # if the file has headers and pay attention to the columns (whether have peptide binding site labels)
		seq, pep, label, pep_ss, seq_ss  = line.strip().split('\t')
		pep_set.add(pep)
		seq_set.add(seq)
		pep_ss_set.add(pep_ss)
		seq_ss_set.add(seq_ss)

	f.close()
	pep_len = [len(pep) for pep in pep_set]
	seq_len = [len(seq) for seq in seq_set]
	pep_ss_len = [len(pep_ss) for pep_ss in pep_ss_set]
	seq_ss_len = [len(seq_ss) for seq_ss in seq_ss_set]

	pep_len.sort()
	seq_len.sort()
	pep_ss_len.sort()
	seq_ss_len.sort()
	pad_pep_len = 50 
	pad_prot_len = seq_len[int(0.8*len(seq_len))-1]
	print 'num of peptides', len(pep_len), 'pad_pep_len', pad_pep_len
	print 'seq_set', len(seq_len), 'pad_prot_len', pad_prot_len
	print 'num of peptide ss', len(pep_ss_len), 'pad_pep_len', pad_pep_len
	print 'seq_ss_set', len(seq_ss_len), 'pad_prot_len', pad_prot_len
	np.save('./preprocessing/pad_pep_len',pad_pep_len)
	np.save('./preprocessing/pad_prot_len',pad_prot_len)
	np.save('./preprocessing/pad_pep_len',pad_pep_len)
	np.save('./preprocessing/_pad_prot_len',pad_prot_len)


	# load raw dense features, the directory dense_feature_dict and proprocessing need to be created first.
	with open('./dense_feature_dict/Protein_pssm_dict') as f: # value: (sequence_length, 20) without sigmoid
		protein_pssm_dict = pickle.load(f)

	with open('./dense_feature_dict/Protein_Intrinsic_dict') as f: # value: (sequence_length, 3): long, short, anchor
		protein_intrinsic_dict = pickle.load(f)

	with open('./dense_feature_dict/Peptide_Intrinsic_dict_v3') as f: # value: (sequence_length, 3): long, short, anchor
		peptide_intrinsic_dict = pickle.load(f)

	peptide_feature_dict = {}
	protein_feature_dict = {}

	peptide_ss_feature_dict = {}
	protein_ss_feature_dict = {}

	peptide_2_feature_dict = {}
	protein_2_feature_dict = {}

	peptide_dense_feature_dict = {}
	protein_dense_feature_dict = {}

	f = open(datafile)
	for line in f.readlines()[1:]:
	        seq, pep, label, pep_ss, seq_ss  = line.strip().split('\t')
		if pep not in peptide_feature_dict:
			feature = label_sequence(pep, pad_pep_len, amino_acid_set)
			peptide_feature_dict[pep] = feature
		if seq not in protein_feature_dict:
			feature = label_sequence(seq, pad_prot_len, amino_acid_set)
			protein_feature_dict[seq] = feature
		if pep_ss not in peptide_ss_feature_dict:
			feature = label_seq_ss(pep_ss, pad_pep_len, seq_ss_dict)
			peptide_ss_feature_dict[pep_ss] = feature
		if seq_ss not in protein_ss_feature_dict:
			feature = label_seq_ss(seq_ss, pad_prot_len, seq_ss_dict)
			protein_ss_feature_dict[seq_ss] = feature
		if pep not in peptide_2_feature_dict:
			feature = label_sequence(pep, pad_pep_len, physicochemical_set)
			peptide_2_feature_dict[pep] = feature
		if seq not in protein_2_feature_dict:
			feature = label_sequence(seq, pad_prot_len, physicochemical_set)
			protein_2_feature_dict[seq] = feature
		if pep not in peptide_dense_feature_dict:
			feature = padding_intrinsic_disorder(peptide_intrinsic_dict[pep], pad_pep_len)
			peptide_dense_feature_dict[pep] = feature
		if seq not in protein_dense_feature_dict:
			feature_pssm = padding_sigmoid_pssm(protein_pssm_dict[seq], pad_prot_len)
			feature_intrinsic = padding_intrinsic_disorder(protein_intrinsic_dict[seq], pad_prot_len)
			feature_dense = np.concatenate((feature_pssm, feature_intrinsic), axis=1)
			protein_dense_feature_dict[seq] = feature_dense
		if seq not in protein_intrinsic_feature_dict:
			feature_intrinsic = padding_intrinsic_disorder(protein_intrinsic_dict[seq], pad_prot_len)
			protein_intrinsic_feature_dict[seq] = feature_intrinsic

	f.close()

	with open('./preprocessing/peptide_feature_dict','wb') as f:
		pickle.dump(peptide_feature_dict,f)
	with open('./preprocessing/protein_feature_dict','wb') as f:
		pickle.dump(protein_feature_dict,f)
	with open('./preprocessing/peptide_ss_feature_dict','wb') as f:
		pickle.dump(peptide_ss_feature_dict,f)
	with open('./preprocessing/protein_ss_feature_dict','wb') as f:
		pickle.dump(protein_ss_feature_dict,f)
	with open('./preprocessing/peptide_2_feature_dict','wb') as f:
		pickle.dump(peptide_2_feature_dict,f)
	with open('./preprocessing/protein_2_feature_dict','wb') as f:
		pickle.dump(protein_2_feature_dict,f)
	with open('./preprocessing/peptide_dense_feature_dict','wb') as f:
		pickle.dump(peptide_dense_feature_dict,f)
	with open('./preprocessing/protein_dense_feature_dict','wb') as f:
	        pickle.dump(protein_dense_feature_dict,f)




