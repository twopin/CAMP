import numpy as np
import pickle

import numpy as np
from scipy.cluster.hierarchy import fcluster, single


# reading sim-mat 
def sequence_clustering(protein_list):
    idx_list = [protein_list.index(pid) for pid in protein_list]


print('start protein/peptide clustering...')
protein_sim_mat = np.load('sim_mat.npy').astype(np.float32)
sim_mat = protein_sim_mat[idx_list, :]
sim_mat = sim_mat[:, idx_list]
print('original sim_mat', protein_sim_mat.shape, 'subset sim_mat', sim_mat.shape)
P_dist = []
for i in range(sim_mat.shape[0]):
    P_dist += (1 - sim_mat[i, (i + 1):]).tolist()
P_dist = np.array(P_dist)
P_link = single(P_dist)
for thre in [0.3, 0.4, 0.5, 0.6, 0.0001]:
    Prot_clusters = fcluster(P_link, thre, 'distance')
    len_list = []
    for i in range(1, max(Prot_clusters) + 1):
        len_list.append(Prot_clusters.tolist().count(i))
    print('thre', thre, 'total num of sequences', len(protein_list), 'num of clusters', max(Prot_clusters),
          'max length', max(len_list))
    Prot_cluster_dict = {protein_list[i]: Prot_clusters[i] for i in range(len(protein_list))}
    with open('prot_cluster_dict_' + str(thre), 'wb') as f:  # or peProt_cluster_dict_
        pickle.dump(Prot_cluster_dict, f, protocol=0)


def split_train_test_clusters(measure, clu_thre, n_fold):
    # load cluster dict
    cluster_path = './'
    with open('Prot_cluster_dict_' + str(clu_thre), 'rb') as f:
        Pep_cluster_dict = pickle.load(f)
    with open('prot_cluster_dict_' + str(clu_thre), 'rb') as f:
        Prot_cluster_dict = pickle.load(f)

    Pep_cluster_set = set(list(Pep_cluster_dict.values()))
    Prot_cluster_set = set(list(Prot_cluster_dict.values()))
    Pep_cluster_list = np.array(list(Pep_cluster_set))
    Prot_cluster_list = np.array(list(Prot_cluster_set))
    np.random.shuffle(Pep_cluster_list)
    np.random.shuffle(Prot_cluster_list)

    pep_kf = KFold(n_fold, shuffle=True)
    prot_kf = KFold(n_fold, shuffle=True)
    pep_train_clusters, pep_test_clusters = [], []
    for train_idx, test_idx in pep_kf.split(Pep_cluster_list):
        pep_train_clusters.append(Pep_cluster_list[train_idx])
        pep_test_clusters.append(Pep_cluster_list[test_idx])
    prot_train_clusters, prot_test_clusters = [], []
    for train_idx, test_idx in prot_kf.split(Prot_cluster_list):
        prot_train_clusters.append(Prot_cluster_list[train_idx])
        prot_test_clusters.append(Prot_cluster_list[test_idx])

    pair_kf = KFold(n_fold, shuffle=True)
    pair_list = []
    for i_c in Pep_cluster_list:
        for i_p in Prot_cluster_list:
            pair_list.append('c' + str(i_c) + 'p' + str(i_p))
    pair_list = np.array(pair_list)
    np.random.shuffle(pair_list)
    # pair_kf = KFold(len(pair_list), n_fold, shuffle=True)
    pair_train_clusters, pair_test_clusters = [], []
    for train_idx, test_idx in pair_kf.split(pair_list):
        pair_train_clusters.append(pair_list[train_idx])
        pair_test_clusters.append(pair_list[test_idx])

    return pair_train_clusters, pair_test_clusters, pep_train_clusters, pep_test_clusters, prot_train_clusters, prot_test_clusters, Pep_cluster_dict, Prot_cluster_dict


def split_data(measure, setting, clu_thre, n_fold, X_pep_seq, X_prot_seq):
    pep_id_list = []
    prot_id_list = []
    pep_id_list = X_pep_seq.tolist()
    prot_id_list = X_prot_seq.tolist()
    n_sample = len(pep_id_list)
    train_idx_list, valid_idx_list, test_idx_list = [], [], []
    print
    'setting:', setting

    elif setting == 'new_protein':
    pair_train_clusters, pair_test_clusters, pep_train_clusters, pep_test_clusters, prot_train_clusters, prot_test_clusters, Pep_cluster_dict, Prot_cluster_dict \
        = split_train_test_clusters(measure, clu_thre, n_fold)
    for fold in range(n_fold):
        prot_train, prot_test = prot_train_clusters[fold], prot_test_clusters[fold]
        prot_train = set(prot_train)
        train_idx, valid_idx, test_idx = [], [], []
        for ele in range(n_sample):
            if Prot_cluster_dict[prot_id_list[ele]] in prot_train:
                train_idx.append(ele)
            elif Prot_cluster_dict[prot_id_list[ele]] in prot_test:
                test_idx.append(ele)
            else:
                print('error')
        train_idx_list.append(train_idx)
        valid_idx = np.random.choice(train_idx, int(len(train_idx) / 10), replace=False)
        valid_idx_list.append(valid_idx)
        test_idx_list.append(test_idx)
        print
        'fold', fold, 'train ', len(train_idx), 'test ', len(test_idx), 'valid ', len(valid_idx)

elif setting == 'new_peptide':
pair_train_clusters, pair_test_clusters, pep_train_clusters, pep_test_clusters, prot_train_clusters, prot_test_clusters, Pep_cluster_dict, Prot_cluster_dict \
    = split_train_test_clusters(measure, clu_thre, n_fold)
for fold in range(n_fold):
    pep_train, pep_test = pep_train_clusters[fold], pep_test_clusters[fold]
    pep_train = set(pep_train)
    train_idx, valid_idx, test_idx = [], [], []
    for ele in range(n_sample):
        if Pep_cluster_dict[pep_id_list[ele]] in pep_train:
            train_idx.append(ele)
        elif Pep_cluster_dict[pep_id_list[ele]] in pep_test:
            test_idx.append(ele)
        else:
            print('error')
    train_idx_list.append(train_idx)
    valid_idx = np.random.choice(train_idx, int(len(train_idx) / 10), replace=False)
    valid_idx_list.append(valid_idx)
    test_idx_list.append(test_idx)
    print
    'fold', fold, 'train ', len(train_idx), 'test ', len(test_idx), 'valid ', len(valid_idx)

elif setting == 'both_new':
assert n_fold ** 0.5 == int(n_fold ** 0.5)
pair_train_clusters, pair_test_clusters, pep_train_clusters, pep_test_clusters, prot_train_clusters, prot_test_clusters, Pep_cluster_dict, Prot_cluster_dict \
    = split_train_test_clusters(measure, clu_thre, int(n_fold ** 0.5))

for fold_x in range(int(n_fold ** 0.5)):
    for fold_y in range(int(n_fold ** 0.5)):
        pep_train, prot_train = pep_train_clusters[fold_x], prot_train_clusters[fold_y]
        pep_test, prot_test = pep_test_clusters[fold_x], prot_test_clusters[fold_y]
        pep_train = set(pep_train)
        prot_train = set(prot_train)

        train_idx, valid_idx, test_idx = [], [], []
        for ele in range(n_sample):
            if Pep_cluster_dict[pep_id_list[ele]] in pep_train and Prot_cluster_dict[prot_id_list[ele]] in prot_train:
                train_idx.append(ele)
            elif Pep_cluster_dict[pep_id_list[ele]] in pep_test and Prot_cluster_dict[prot_id_list[ele]] in prot_test:
                test_idx.append(ele)
        train_idx_list.append(train_idx)
        valid_idx = np.random.choice(train_idx, int(len(train_idx) / 10), replace=False)
        valid_idx_list.append(valid_idx)
        test_idx_list.append(test_idx)
        print
        'fold', fold_x * int(n_fold ** 0.5) + fold_y, 'train ', len(train_idx), 'test ', len(test_idx), 'valid ', len(
            valid_idx)
return train_idx_list, valid_idx_list, test_idx_list
