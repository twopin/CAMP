from keras.layers import *
from keras.models import *
from keras.optimizers import rmsprop
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from  Self_Attention import *
import math, sys, sklearn, pickle
import numpy as np
import argparse as ap
import tensorflow as tf
from camp_utils import *

batch_size = 256 
pad_pep_len = 50
pad_seq_len = int(np.load('./preprocessing/pad_seq_len.npy'))


def get_session(gpu_fraction=0.9):
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
	return tf.Session(config=config)

set_session(get_session())

def binding_vec_pos(bs_str,N):
	if bs_str == 'NoBinding':
		print('Error! This record is positive.')
		return None
	if bs_str == '-99999':
		bs_vec = np.zeros(N)
		bs_vec.fill(-99999)
                return bs_vec
	else:
		bs_list = [int(x) for x in bs_str.split(',')]
		bs_list = [x for x in bs_list if x<N]
		bs_vec = np.zeros(N)
		bs_vec[bs_list]=1

		return bs_vec

def binding_vec_neg(bs_str,N):
	if bs_str!= 'NoBinding':
		print('Error! This record is negative.')
		return None
	else :
		bs_vec = np.zeros(N)
		return bs_vec


def get_mask(protein_seq,pad_seq_len):
	if len(protein_seq)<=pad_seq_len:
		a = np.zeros(pad_seq_len)
		a[:len(protein_seq)] = 1
	else:
		cut_protein_seq = protein_seq[:pad_seq_len]
		a = np.zeros(pad_seq_len)
		a[:len(cut_protein_seq)] = 1
	return a


# flag is an indicator for checking whether this record has binding sites information
def boost_mask_BCE_loss(input_mask,flag):
	def conditional_BCE(y_true, y_pred):
                loss = flag * K.binary_crossentropy(y_true, y_pred) * input_mask
		return K.sum(loss) / K.sum(input_mask)
	return conditional_BCE



def load_test(name):
	print('loading feature:'), name
	
	with open('./preprocessing/protein_feature_dict') as f: 
		protein_feature_dict = pickle.load(f)
	with open('./preprocessing/peptide_feature_dict') as f:
		peptide_feature_dict = pickle.load(f)
	with open('./preprocessing/protein_ss_feature_dict') as f: 
		protein_ss_feature_dict = pickle.load(f)
	with open('./preprocessing/peptide_ss_feature_dict') as f:
		peptide_ss_feature_dict = pickle.load(f)
	with open('./preprocessing/protein_2_feature_dict') as f: 
		protein_2_feature_dict = pickle.load(f)
	with open('./preprocessing/peptide_2_feature_dict') as f:
		peptide_2_feature_dict = pickle.load(f)
	with open('./preprocessing/protein_dense_feature_dict') as f: 
		protein_dense_feature_dict = pickle.load(f)
	with open('./preprocessing/peptide_dense_feature_dict') as f:
		peptide_dense_feature_dict = pickle.load(f)
	with open('./preprocessing/protein_pssm_feature_dict') as f: 
		protein_pssm_feature_dict = pickle.load(f)
	with open('./preprocessing/protein_intrinsic_feature_dict') as f: 
		protein_intrinsic_feature_dict = pickle.load(f)

	
	datafile = 'predict_input_datafile'
	print('load feature dict')
	X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p = [], [], [], [], [], []
	X_dense_pep,X_dense_p,X_pssm_p,X_intrinsic_p = [],[],[],[]
	pep_sequence, prot_sequence, Y = [], [], []
	Y_pep_bs, X_pep_mask, X_bs_flag = [], [], []

	with open(datafile) as f:
		for line in f.readlines()[1:]:
			seq, peptide, peptide_ss, seq_ss, pep_bs  = line.strip().split('\t')
			if int(label)==1:
				pep_bs_vec = binding_vec_pos(pep_bs,pad_pep_len)
				if pep_bs == '-99999':
					flag =0.0
				else : 
					flag =1.0
			if int(label)==0:
				flag==0.0
				pep_bs_vec = binding_vec_neg(pep_bs,pad_pep_len)

			X_pep_mask.append(get_mask(peptide,pad_pep_len))
			Y_pep_bs.append(pep_bs_vec)
			X_bs_flag.append(flag)

			pep_sequence.append(peptide)
			prot_sequence.append(seq)
			Y.append(label)
			X_pep.append(peptide_feature_dict[peptide])
			X_p.append(protein_feature_dict[seq])
			X_SS_pep.append(peptide_ss_feature_dict[peptide_ss])
			X_SS_p.append(protein_ss_feature_dict[seq_ss])
			X_2_pep.append(peptide_2_feature_dict[peptide])
			X_2_p.append(protein_2_feature_dict[seq])
			X_dense_pep.append(peptide_dense_feature_dict[peptide])
			X_dense_p.append(protein_dense_feature_dict[seq])
			X_pssm_p.append(protein_pssm_feature_dict[seq])
			X_intrinsic_p.append(protein_intrinsic_feature_dict[seq])
			
	X_pep = np.array(X_pep)
	X_p = np.array(X_p)
	X_SS_pep = np.array(X_SS_pep)
	X_SS_p = np.array(X_SS_p)
	X_2_pep = np.array(X_2_pep)
	X_2_p = np.array(X_2_p)
	X_dense_pep = np.array(X_dense_pep)
	X_dense_p = np.array(X_dense_p)
	X_pssm_p = np.array(X_pssm_p)
	X_intrinsic_p = np.array(X_intrinsic_p)
	Y = np.array(Y).astype(int)

	X_pep_mask = np.array(X_pep_mask)
	Y_pep_bs = np.array(Y_pep_bs)
	X_bs_flag = np.array(X_bs_flag)

	pep_sequence = np.array(pep_sequence)
	prot_sequence = np.array(prot_sequence)
	train_idx = range(Y.shape[0])
	np.random.shuffle(train_idx)
	X_pep = X_pep[train_idx]
	X_p = X_p[train_idx]
	X_SS_pep = X_SS_pep[train_idx]
	X_SS_p = X_SS_p[train_idx]
	X_2_pep = X_2_pep[train_idx]
	X_2_p = X_2_p[train_idx]
	X_dense_pep = X_dense_pep[train_idx]
	X_dense_p = X_dense_p[train_idx]
	X_pssm_p = X_pssm_p[train_idx]
	X_intrinsic_p = X_intrinsic_p[train_idx]
	Y = Y[train_idx]

	X_pep_mask = X_pep_mask[train_idx]
	X_bs_flag = X_bs_flag[train_idx]
	Y_pep_bs = Y_pep_bs[train_idx]
	
	pep_sequence = pep_sequence[train_idx]
	prot_sequence = prot_sequence[train_idx]
	
	return X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, X_pssm_p, X_intrinsic_p, Y, pep_sequence, prot_sequence, X_pep_mask, Y_pep_bs, X_bs_flag



model_name='./model/CAMP_BS.h5'
print('Start loading model :', model_name)
model = load_model(model_name,custom_objects={'Self_Attention': Self_Attention,'boost_mask_BCE_loss':boost_mask_BCE_loss})
#model.summary()
print('Finish loading CAMP.')
X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, X_pssm_p, X_intrinsic_p, Y, \
pep_sequence, prot_sequence, X_pep_mask, Y_pep_bs, X_bs_flag = load_test(name)

print('Start predicting .test_xc shape is',np.array(X_pep).shape)
pred_label = model.predict([np.array(X_pep),np.array(X_p),np.array(X_SS_pep),np.array(X_SS_p),\
							np.array(X_2_pep),np.array(X_2_p),np.array(X_dense_pep),np.array(X_dense_p),\
							np.array(X_pep_mask),np.array(X_bs_flag)],batch_size=batch_size)[0]
pred_bs = model.predict([np.array(X_pep),np.array(X_p),np.array(X_SS_pep),np.array(X_SS_p),\
							np.array(X_2_pep),np.array(X_2_p),np.array(X_dense_pep),np.array(X_dense_p),\
							np.array(X_pep_mask),np.array(X_bs_flag)],batch_size=batch_size)[1]

np.save('./bs_pred_test',np.array(pred_label))
np.save('./bs_test_Y',np.array(Y))
np.save('./bs_test_peptide',np.array(pep_sequence))
np.save('./pred_Y_pep_bs',np.array(pred_bs))
np.save('./bs_test_protein',np.array(prot_sequence))

print('Finish predicting and reach end point for this prediction.')

del model


