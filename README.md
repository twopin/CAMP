# CAMP: a Convolutional Attention-based Neural Network for Multifaceted Peptide-protein Interaction Prediction

CAMP is a sequence-based deep learning framework for multifaceted prediction of peptide-protein interactions, including not only binary peptide-protein interactions, but also corresponding peptide binding residues.

### Notice

Since CAMP exploits a series of sequence-based features, you CANNOT use CAMP when the primary sequence information of peptides and proteins is unknown. In addition, you need to replace any non-standard amino acid with 'X' first.

### Requirment

Python2.7, Keras=2.0.8, Tensorflow=1.2.1, RDKit (for data preprocessing), CUDA (GPU computation)

### Running CAMP

1. To test the code, use the test data with the command `python -u train_camp.py ./test_data`.
2. To reproduce the data described in the paper, please refer to the instructuion in Data Preparation. 


### Train CAMP from scratch

Here we offer the protocol and some sample data. Due to the copyright issues, we can not provide the complete benchmark dataset. You may follow the procedures in ./data/ to reproduce the benchmark dataset. 

- STEP 0: Downloading data from three sources (skip this step if you have your own data)

1. RCSB PDB : Download the fasta files from ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz and pdb files 
	
2. UniProt : Download pdb_chain_uniprot.tsv.gz containing the UniProt IDs of proteins in PDB complexes, which will be used to map the UniProt sequences later from https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html

3. DrugBank : Download full database.xml from DrugBank, to access the data, you need to create an account first. (requiring licenses)

- STEP 1: Process PDB data

### Predict novel pairs

To predict your own data, you need to have the amino acid sequences of the peptide-protein pairs. Then you need to generate the corresponding features (you can either use the online servers or download the softwares):

- STEP 0: Generated sequnece-based features

1. Secondary structure : http://scratch.proteomics.ics.uci.edu/explanation.html#SSpro
2. Intrinsic Disorder :  https://iupred2a.elte.hu/
3. PSSM matrix: ftp://ftp.cnbi.nlm.nih.gov/blast/executables/blast+/LATEST/
4. Use the functions in ./features/step3_generate_features.py to process raw output files. Then put all feature dicts in ./dense_feature_dict/

- STEP 1: Proprecess data with feature file
1. format the pepitde-protein data like (protein sequence, peptide sequence, protein_ss, peptide_ss)
2. Use the command 'python preprocess_features.py' to obtain processed feature dicts in ./preprocessing/

- STEP 2: Make predictions by the CAMP model
To predict binary peptide-protein interactions with corresponding peptide binding residues, use the command `python -u predict_camp.py`. 
