import numpy as np
import pandas as pd


def check_abnormal_aa(peptide_seq):
    len_seq = len(peptide_seq)
    cnt = 0
    standard_aa = [
        "G",
        "A",
        "P",
        "V",
        "L",
        "I",
        "M",
        "F",
        "Y",
        "W",
        "S",
        "T",
        "C",
        "N",
        "Q",
        "K",
        "H",
        "R",
        "D",
        "E",
    ]
    for i in peptide_seq:
        if i in standard_aa:
            cnt = cnt + 1
    score = float(cnt) / len_seq
    return score


def lower_chain(input_str):
    chain_list = list(input_str)
    output_list = []

    for item in chain_list:
        if item.isalpha():
            a = item.lower()
        else:
            a = item
        output_list.append(a)
    output_str = "".join(output_list)
    return output_str


# Step 0: parse the fasta file downloaded from the RCSB PDB
# INPUT : pdb_seqres.txt
# OUTPUT: pdb_pep_chain, pdbid_all_fasta
raw_str = ""
with open("pdb_seqres_test.txt", "r") as f:
    for line in f.readlines():
        raw_str = raw_str + line.replace("\n", "###")
raw_list = raw_str.split(">")
del raw_list[0]

PDB_id_lst = [x.split("_")[0] for x in raw_list]
PDB_chain_lst = [x.split("_")[1].split(" ")[0].lower() for x in raw_list]
PDB_type_lst = [x.split("mol:")[1].split(" ")[0] for x in raw_list]
PDB_seq_lst = [x.split("###")[1] for x in raw_list]
PDB_seq_len_lst = [len(x) for x in PDB_seq_lst]
df_fasta_raw = pd.DataFrame(
    list(zip(PDB_type_lst, PDB_seq_len_lst, PDB_seq_lst, PDB_id_lst, PDB_chain_lst)),
    columns=["PDB_type", "PDB_seq_len", "PDB_seq", "PDB_id", "chain"],
)
df_fasta = df_fasta_raw[
    (df_fasta_raw.PDB_seq_len <= 50) & (df_fasta_raw.PDB_type == "protein")
]
df_fasta_raw.to_csv("pdbid_all_fasta", encoding="utf-8", index=False, sep="\t")
df_fasta.to_csv("pdb_pep_chain", encoding="utf-8", index=False, sep="\t")

print("Step 0 is finished by generating two files : pdb_pep_chain & pdbid_all_fasta!")


# Step1 : Load all PDB ids that might contain peptide interaction and plip prediction results
# INPUT : pdb_pep_chain from Step 0 & analyzed file generated by PLIP (placed under ./peptide_result/). There is an example of PLIP result file called example_PLIP_result.txt)
# OUTPUT: plip_predict_result


def load(pdb_pep_dataset, plip_result_filename):  # pdb_pep_chain   #plip_predict_result
    df_fasta_pep = pd.read_csv(pdb_pep_dataset, sep="\t", header=0)
    df_fasta_pep = df_fasta_pep.reset_index(drop=True)
    df_predict = pd.DataFrame(columns=["pdb_id", "pep_chain", "predicted_chain"])

    for i in range(df_fasta_pep.shape[0]):
        pdb_id = df_fasta_pep["PDB_id"][i]
        chain = df_fasta_pep["chain"][i]
        result_file_name = "./peptide_result/" + pdb_id + "_" + chain + "_result.txt"
        # print(result_file_name)
        try:
            for line in open(result_file_name):
                if line.startswith("Interacting chain(s):"):
                    df_predict.loc[i] = [
                        pdb_id,
                        chain,
                        str(line)
                        .replace("\n", "")
                        .replace("\r", "")
                        .replace("Interacting chain(s):", "")
                        .lower(),
                    ]
            if i % 5000 == 0:
                print("already finished files", i)
        except:
            # print('find no file for',pdb_id)
            # print(i,pdb_id,line)
            pass
    print("finish loading!")
    print("-----------------------------------------------------")
    # print(df_predict.info())
    df_predict["predicted_chain_num"] = df_predict.predicted_chain.apply(
        lambda x: len(x.replace(" ", ""))
    )
    df_predict = df_predict.loc[df_predict.predicted_chain_num > 0]
    df_predict = df_predict.drop("predicted_chain_num", axis=1)
    df_predict["predicted_chain"] = df_predict.predicted_chain.apply(
        lambda x: x.replace(" ", "")
    )
    df_predict["pep_chain"] = df_predict.pep_chain.apply(lambda x: x.replace(" ", ""))
    df_predict = df_predict.reset_index(drop=True)
    print("finish removing PDB ids without any interaction")
    print("-----------------------------------------------------")
    df_predict.predicted_chain = df_predict.predicted_chain.apply(
        lambda x: x.split(",")
    )
    lst_col = "predicted_chain"
    df1 = pd.DataFrame(
        {
            col: np.repeat(df_predict[col].values, df_predict[lst_col].str.len())
            for col in df_predict.columns.difference([lst_col])
        }
    ).assign(**{lst_col: np.concatenate(df_predict[lst_col].values)})[
        df_predict.columns.tolist()
    ]
    df_predict = df1
    # save organized data formatted like (pdb,pep_chain,predicted_prot_chain)
    file_name = plip_result_filename
    df_predict.to_csv(file_name, encoding="utf-8", index=False, sep="\t")
    print(
        "finish exploding comma-seperated predicted chain, successfully saved records:",
        df_predict.shape[0],
    )

    print(
        "Step 1 is finished by generating the PLIP prediction file : plip_predict_result. "
    )

    return df_predict


# Step 2: Get fasta sequence of the predicted interacting chains
# INPUT:  pdbid_all_fasta from Step 0
# OUTPUT: -
def load_all_fasta(all_fasta_file, input_dataset):  # pdbid_all_fasta # df_predict
    df_fasta = pd.read_csv(all_fasta_file, sep="\t", header=0)
    df_fasta_protein = df_fasta.loc[df_fasta.PDB_type == "protein"]

    df_fasta_protein["PDB_id"] = df_fasta_protein.PDB_id_chain.apply(
        lambda x: x.split("_")[0]
    )
    df_fasta_protein["chain"] = df_fasta_protein.PDB_id_chain.apply(
        lambda x: x.split("_")[1].lower()
    )
    df_fasta_vocabulary = df_fasta_protein[["PDB_id", "chain", "PDB_seq"]]

    df_predict_det = pd.merge(
        input_dataset,
        df_fasta_vocabulary,
        how="left",
        left_on=["pdb_id", "pep_chain"],
        right_on=["PDB_id", "chain"],
    )

    df_predict_det1 = pd.merge(
        df_predict_det,
        df_fasta_vocabulary,
        how="left",
        left_on=["pdb_id", "predicted_chain"],
        right_on=["PDB_id", "chain"],
    )
    df_predict_det1 = df_predict_det1.drop(
        ["PDB_id_x", "chain_x", "PDB_id_y", "chain_y"], axis=1
    )
    df_predict_det1.columns = [
        "pdb_id",
        "pep_chain",
        "predicted_chain",
        "pep_seq",
        "prot_seq",
    ]
    df_predict_det1["pep_seq_len"] = df_predict_det1.pep_seq.apply(lambda x: len(x))
    df_predict_det1["prot_seq_len"] = df_predict_det1.prot_seq.apply(lambda x: len(x))

    # check sequence length(peptide<=50 & protein >50)
    df_predict_det1 = df_predict_det1.loc[
        (df_predict_det1.pep_seq_len <= 50) & (df_predict_det1.prot_seq_len > 50)
    ]

    # remove records with more than 20% AA is abnormal
    df_predict_det1["peptide_seq_score"] = df_predict_det1.pep_seq.apply(
        lambda x: check_abnormal_aa(x)
    )
    df_predict_det1 = df_predict_det1[df_predict_det1.peptide_seq_score >= 0.8]

    print("finish removing sequences without too many non-standard residues")
    print("-----------------------------------------------------")

    return df_predict_det1


# Step 3: Map Uniprot ID for PDB complex by protein-chain & PDB id
# INPUT: data from Step 2 & pdb_chain_uniprot.tsv from SIFT
# OUTPUT: UniProt_ID_list ( all IDs are the searching query on https://www.uniprot.org/uploadlists/ for unified sequence)
def map_uniprot_chain(
    input_dataset, pdb_chain_uniprot_file
):  # df_predict_det1 #pdb_chain_uniprot.tsv
    df_sifts = pd.read_csv(pdb_chain_uniprot_file, sep="\t", header=0)
    df_sifts = df_sifts[["PDB", "CHAIN", "SP_PRIMARY"]]
    df_sifts_keep = df_sifts[df_sifts["CHAIN"] != df_sifts["CHAIN"]]
    df_sifts = df_sifts[df_sifts["CHAIN"] == df_sifts["CHAIN"]]
    df_sifts["CHAIN"] = df_sifts.CHAIN.apply(lambda x: lower_chain(x))

    df_predict_det2 = pd.merge(
        input_dataset,
        df_sifts,
        how="left",
        left_on=["pdb_id", "predicted_chain"],
        right_on=["PDB", "CHAIN"],
    )
    df_predict_det2 = df_predict_det2.drop(["PDB", "CHAIN"], axis=1)

    # subset records that don't have a matched protein chain Uniprot
    df_predict_det2_no_uni = df_predict_det2[
        df_predict_det2.SP_PRIMARY != df_predict_det2.SP_PRIMARY
    ]
    df_predict_det2_no_uni = df_predict_det2_no_uni.reset_index(drop=True)

    df_predict_det2_no_uni = df_predict_det2_no_uni.drop(
        ["prot_seq_len", "peptide_seq_score"], axis=1
    )
    df_predict_det2_no_uni = df_predict_det2_no_uni[
        [
            "pdb_id",
            "pep_chain",
            "predicted_chain",
            "pep_seq",
            "pep_seq_len",
            "SP_PRIMARY",
            "prot_seq",
        ]
    ]
    df_predict_det2_no_uni.rename(columns={"prot_seq": "Sequence"}, inplace=True)

    # focus on records with Uniprot Ids
    df_predict_det3 = df_predict_det2[
        df_predict_det2.SP_PRIMARY == df_predict_det2.SP_PRIMARY
    ]

    # save matched uniport ID for retrieving information from Uniprot Website
    df_uni_id = df_predict_det3[["SP_PRIMARY"]]
    df_uni_id.drop_duplicates(inplace=True)
    file_name = pdb_chain_uniprot_file
    df_uni_id.to_csv(file_name, encoding="utf-8", index=False, sep="\t")

    return df_predict_det2_no_uni, df_predict_det3


# Step 4: Load Uniport sequences and family information & filter out MHC families
# INPUT: the data from Step 3 & uniprot2seq from UniProt Website (a tab separated file with fields including Uniprot_id,Uniprot Sequence,Protein_name,Protein_families)
# OUTPUT: interacted peptide-protein pairs from PDB (a '#' separated file with fields including pdb_id,pep_chain,prot_chain,pep_seq,Uniprot_id,prot_seq,protein_families)


def load_uni_seq(input_dataset, uniprot2seq_file):
    df_uni2seq = pd.read_csv(uniprot2seq_file, sep="\t", header=0)
    df_uni2seq = df_uni2seq.drop("uniprot", axis=1)
    df_uni2seq = df_uni2seq.drop_duplicates(["Uniprot_id", "Sequence"], keep="first")
    df_uni2seq = df_uni2seq.fillna("Unknown_from_uniprot")

    # join by uniprot id
    df_predict_det4 = pd.merge(
        input_dataset,
        df_uni2seq,
        how="left",
        left_on=["SP_PRIMARY"],
        right_on=["Uniprot_id"],
    )
    df_predict_det4 = df_predict_det4.drop(
        ["Uniprot_id", "Protein_name", "prot_seq", "prot_seq_len", "peptide_seq_score"],
        axis=1,
    )
    df_predict_det4 = df_predict_det4.drop_duplicates(
        ["pdb_id", "pep_seq", "SP_PRIMARY", "Sequence"], keep="first"
    )

    # filter out MHC
    df_predict_det4["MHC_flag"] = df_predict_det4.Protein_families.apply(
        lambda x: x.lower().find("mhc")
    )
    df_mhc = df_predict_det4.loc[df_predict_det4.MHC_flag != -1][
        ["pdb_id", "Protein_families"]
    ]
    df_mhc.columns = ["pdb_id_mhc", "prot_family_mhc"]

    # join by  PDB id only(if a pdb contains mhc proteins,remove all records of the PDB id)
    df_predict_det5 = pd.merge(
        df_predict_det4, df_mhc, left_on=["pdb_id"], right_on=["pdb_id_mhc"], how="left"
    )
    df_predict_det5 = df_predict_det5.loc[
        df_predict_det5.pdb_id_mhc != df_predict_det5.pdb_id_mhc
    ]
    df_predict_det5 = df_predict_det5.drop(
        ["pdb_id_mhc", "prot_family_mhc", "MHC_flag"], axis=1
    )
    df_predict_det5.drop_duplicates(inplace=True)

    df_predict_det2_no_uni = df_predict_det2_no_uni.drop(
        ["prot_seq_len", "peptide_seq_score"], axis=1
    )
    df_predict_det2_no_uni = df_predict_det2_no_uni[
        [
            "pdb_id",
            "pep_chain",
            "predicted_chain",
            "pep_seq",
            "pep_seq_len",
            "SP_PRIMARY",
            "prot_seq",
        ]
    ]
    df_predict_det2_no_uni.rename(columns={"prot_seq": "Sequence"}, inplace=True)
    df_predict_det2_no_uni["Protein_families"] = pd.Series(
        ["Unknown Uniprot_ids" for x in range(df_predict_det2_no_uni.shape[0])]
    )
    df_predict_det6 = pd.concat(
        [df_predict_det2_no_uni, df_predict_det5], ignore_index=True
    )
    df_predict_det6["plip_prot_chain"] = df_predict_det6.predicted_chain.apply(
        lambda x: x.upper()
    )
    df_predict_det6 = df_predict_det6.drop_duplicates(
        ["pep_seq", "Sequence"], keep="first"
    )
    df_predict_det6 = df_predict_det6.reset_index(drop=True)  # 8184
    df_predict_det6["prot_seq_len"] = df_predict_det6.Sequence.apply(
        lambda x: len(str(x))
    )
    df_predict_det6 = df_predict_det6[df_predict_det6.prot_seq_len <= 5000]

    df_pdb_pairs = df_predict_det6[
        [
            "pdb_id",
            "pep_chain",
            "plip_prot_chain",
            "pep_seq",
            "SP_PRIMARY",
            "Sequence",
            "Protein_families",
        ]
    ]
    df_pdb_pairs.columns = [
        "pdb_id",
        "pep_chain",
        "prot_chain",
        "pep_seq",
        "Uniprot_id",
        "prot_seq",
        "protein_families",
    ]
    file_name = "train_pairs_pdb"
    df_pdb_pairs.to_csv(file_name, encoding="utf-8", index=False, sep="#")

    return df_pdb_pairs


df_predict = load("pdb_pep_chain", "plip_predict_result")
df_predict_det1 = load_all_fasta("pdbid_all_fasta", df_predict)
df_predict_det2_no_uni, df_predict_det3 = map_uniprot_chain(
    df_predict_det1, "pdb_chain_uniprot.tsv", "uniport_id_list"
)
df_pdb_pairs = load_uni_seq(df_predict_det3, "uniprot2seq")
