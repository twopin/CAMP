from optparse import OptionParser

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import *
from keras.models import *

from Self_Attention import *
from camp_utils import *

parser = OptionParser()
parser.add_option("-b", "--b", default=256, help="The batch size b")
parser.add_option("-l", "--l", default=50, help="The padding length for peptides l")
parser.add_option(
    "-f", "--f", default=0.9, help="The fraction of GPU memory to process the model f"
)
parser.add_option(
    "-m",
    "--m",
    default=1,
    help="Using the binary model m=1 or the multi-level model m=2",
)
parser.add_option(
    "-p",
    "--p",
    default=800,
    help="The padding length for proteins p, the length is set to default value when predicting",
)
(opts, args) = parser.parse_args()


def get_session(gpu_fraction):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    return tf.Session(config=config)


def binding_vec_pos(bs_str, N):
    if bs_str == "NoBinding":
        print("Error! This record is positive.")
        return None
    if bs_str == "-99999":
        bs_vec = np.zeros(N)
        bs_vec.fill(-99999)
        return bs_vec
    else:
        bs_list = [int(x) for x in bs_str.split(",")]
        bs_list = [x for x in bs_list if x < N]
        bs_vec = np.zeros(N)
        bs_vec[bs_list] = 1

        return bs_vec


def binding_vec_neg(bs_str, N):
    if bs_str != "NoBinding":
        print("Error! This record is negative.")
        return None
    else:
        bs_vec = np.zeros(N)
        return bs_vec


def get_mask(protein_seq, pad_seq_len):
    if len(protein_seq) <= pad_seq_len:
        a = np.zeros(pad_seq_len)
        a[: len(protein_seq)] = 1
    else:
        cut_protein_seq = protein_seq[:pad_seq_len]
        a = np.zeros(pad_seq_len)
        a[: len(cut_protein_seq)] = 1
    return a


# flag is an indicator for checking whether this record has binding sites information
def boost_mask_BCE_loss(input_mask, flag):
    def conditional_BCE(y_true, y_pred):
        loss = flag * K.binary_crossentropy(y_true, y_pred) * input_mask
        return K.sum(loss) / K.sum(input_mask)

    return conditional_BCE


batch_size = int(opts.b)
pad_pep_len = int(opts.l)
pad_seq_len = int(opts.p)
gpu_frac = int(opts.f)
model_mode = int(opts.m)

set_session(get_session(gpu_frac))


def load_example(model_mode):
    print("loading features:")

    with open("./example_data_feature/protein_feature_dict") as f:
        protein_feature_dict = pickle.load(f)
    with open("./example_data_feature/peptide_feature_dict") as f:
        peptide_feature_dict = pickle.load(f)
    with open("./example_data_feature/protein_ss_feature_dict") as f:
        protein_ss_feature_dict = pickle.load(f)
    with open("./example_data_feature/peptide_ss_feature_dict") as f:
        peptide_ss_feature_dict = pickle.load(f)
    with open("./example_data_feature/protein_2_feature_dict") as f:
        protein_2_feature_dict = pickle.load(f)
    with open("./example_data_feature/peptide_2_feature_dict") as f:
        peptide_2_feature_dict = pickle.load(f)
    with open("./example_data_feature/protein_dense_feature_dict") as f:
        protein_dense_feature_dict = pickle.load(f)
    with open("./example_data_feature/peptide_dense_feature_dict") as f:
        peptide_dense_feature_dict = pickle.load(f)

    datafile = "example_data.tsv"
    print("load feature dict")
    X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p = [], [], [], [], [], []
    X_dense_pep, X_dense_p = [], []
    pep_sequence, prot_sequence, Y = [], [], []
    X_pep_mask, X_bs_flag = [], []

    with open(datafile) as f:
        for line in f.readlines()[1:]:
            seq, peptide, peptide_ss, seq_ss = line.strip().split("\t")
            # if int(label)==1:
            # 	pep_bs_vec = binding_vec_pos(pep_bs,pad_pep_len)
            # 	if pep_bs == '-99999':
            # 		flag =0.0
            # 	else :
            # 		flag =1.0
            # if int(label)==0:
            # 	flag==0.0
            # 	pep_bs_vec = binding_vec_neg(pep_bs,pad_pep_len)
            flag = 1.0  # For prediction, set flag==1 to generate binding sites
            X_pep_mask.append(get_mask(peptide, pad_pep_len))
            X_bs_flag.append(flag)

            pep_sequence.append(peptide)
            prot_sequence.append(seq)
            X_pep.append(peptide_feature_dict[peptide])
            X_p.append(protein_feature_dict[seq])
            X_SS_pep.append(peptide_ss_feature_dict[peptide_ss])
            X_SS_p.append(protein_ss_feature_dict[seq_ss])
            X_2_pep.append(peptide_2_feature_dict[peptide])
            X_2_p.append(protein_2_feature_dict[seq])
            X_dense_pep.append(peptide_dense_feature_dict[peptide])
            X_dense_p.append(protein_dense_feature_dict[seq])

    X_pep = np.array(X_pep)
    X_p = np.array(X_p)
    X_SS_pep = np.array(X_SS_pep)
    X_SS_p = np.array(X_SS_p)
    X_2_pep = np.array(X_2_pep)
    X_2_p = np.array(X_2_p)
    X_dense_pep = np.array(X_dense_pep)
    X_dense_p = np.array(X_dense_p)

    X_pep_mask = np.array(X_pep_mask)
    X_bs_flag = np.array(X_bs_flag)

    pep_sequence = np.array(pep_sequence)
    prot_sequence = np.array(prot_sequence)
    train_idx = range(X_pep.shape[0])
    np.random.shuffle(train_idx)
    X_pep = X_pep[train_idx]
    X_p = X_p[train_idx]
    X_SS_pep = X_SS_pep[train_idx]
    X_SS_p = X_SS_p[train_idx]
    X_2_pep = X_2_pep[train_idx]
    X_2_p = X_2_p[train_idx]
    X_dense_pep = X_dense_pep[train_idx]
    X_dense_p = X_dense_p[train_idx]

    X_pep_mask = X_pep_mask[train_idx]
    X_bs_flag = X_bs_flag[train_idx]

    pep_sequence = pep_sequence[train_idx]
    prot_sequence = prot_sequence[train_idx]

    if model_mode == 1:
        return (
            X_pep,
            X_p,
            X_SS_pep,
            X_SS_p,
            X_2_pep,
            X_2_p,
            X_dense_pep,
            X_dense_p,
            pep_sequence,
            prot_sequence,
        )
    else:
        return (
            X_pep,
            X_p,
            X_SS_pep,
            X_SS_p,
            X_2_pep,
            X_2_p,
            X_dense_pep,
            X_dense_p,
            pep_sequence,
            prot_sequence,
            X_pep_mask,
            X_bs_flag,
        )


if model_mode == 1:
    model_name = "./model/CAMP.h5"
    print("Start loading model :", model_name)
    model = load_model(model_name, custom_objects={"Self_Attention": Self_Attention})
    # model.summary()
    print("Finish loading CAMP.")
    (
        X_pep,
        X_p,
        X_SS_pep,
        X_SS_p,
        X_2_pep,
        X_2_p,
        X_dense_pep,
        X_dense_p,
        pep_sequence,
        prot_sequence,
    ) = load_example(model_mode)

    print("Start predicting.")
    pred_label = model.predict(
        [
            np.array(X_pep),
            np.array(X_p),
            np.array(X_SS_pep),
            np.array(X_SS_p),
            np.array(X_2_pep),
            np.array(X_2_p),
            np.array(X_dense_pep),
            np.array(X_dense_p),
        ],
        batch_size=batch_size,
    )

    np.save("./example_prediction/bs_pred_test", np.array(pred_label))
    np.save("./example_prediction/bs_test_peptide", np.array(pep_sequence))
    np.save("./example_prediction/bs_test_protein", np.array(prot_sequence))

else:
    model_name = "./model/CAMP_BS.h5"
    print("Start loading model :", model_name)
    model = load_model(
        model_name,
        custom_objects={
            "Self_Attention": Self_Attention,
            "boost_mask_BCE_loss": boost_mask_BCE_loss,
        },
    )
    # model.summary()
    print("Finish loading CAMP.")
    (
        X_pep,
        X_p,
        X_SS_pep,
        X_SS_p,
        X_2_pep,
        X_2_p,
        X_dense_pep,
        X_dense_p,
        pep_sequence,
        prot_sequence,
        X_pep_mask,
        X_bs_flag,
    ) = load_example(model_mode)

    print("Start predicting .test_xc shape is", np.array(X_pep).shape)
    pred_label = model.predict(
        [
            np.array(X_pep),
            np.array(X_p),
            np.array(X_SS_pep),
            np.array(X_SS_p),
            np.array(X_2_pep),
            np.array(X_2_p),
            np.array(X_dense_pep),
            np.array(X_dense_p),
            np.array(X_pep_mask),
            np.array(X_bs_flag),
        ],
        batch_size=batch_size,
    )[0]
    pred_bs = model.predict(
        [
            np.array(X_pep),
            np.array(X_p),
            np.array(X_SS_pep),
            np.array(X_SS_p),
            np.array(X_2_pep),
            np.array(X_2_p),
            np.array(X_dense_pep),
            np.array(X_dense_p),
            np.array(X_pep_mask),
            np.array(X_bs_flag),
        ],
        batch_size=batch_size,
    )[1]

    np.save("./example_prediction/bs_pred_test", np.array(pred_label))
    np.save("./example_prediction/bs_test_peptide", np.array(pep_sequence))
    np.save("./example_prediction/pred_pep_bindingsites", np.array(pred_bs))
    np.save("./example_prediction/bs_test_protein", np.array(prot_sequence))

print("Finish predicting and reach end point for this prediction.")

del model
