# CAMP: a Convolutional Attention-based Neural Network for Multifaceted Peptide-protein Interaction Prediction

CAMP is a sequence-based deep learning framework for multifaceted prediction of peptide-protein interactions, including not only binary peptide-protein interactions, but also corresponding peptide binding residues.

### Notice

Since CAMP exploits a series of sequence-based features, you CANNOT use CAMP when the primary sequence information of peptides and proteins is unknown. In addition, you need to replace any non-standard amino acid with 'X' first.

### Requirement

Python2.7, Keras=2.0.8, Tensorflow=1.2.1, RDKit (for data preprocessing), CUDA (GPU computation)

### Running CAMP

1. To test the code, use the test data with the command `python -u train_CAMP.py ./test/example_data`.
2. To reproduce the data described in the paper, please refer to the instructuion in "Data curation".


### Predict novel pairs

To predict your own data, you need to have the amino acid sequences of the peptide-protein pairs. Then you need to generate the corresponding features (you can either use the online servers or download the softwares):

- STEP 0: Generated sequnece-based features

1. Secondary structure : http://scratch.proteomics.ics.uci.edu/explanation.html#SSpro
2. Intrinsic Disorder :  https://iupred2a.elte.hu/
3. PSSM matrix: ftp://ftp.cnbi.nlm.nih.gov/blast/executables/blast+/LATEST/
4. Use the functions in ./data_prepare/step3_generate_features.py to process raw output files. Then put all feature dicts in ./dense_feature_dict/

- STEP 1: Proprecess data with feature file
1. format the pepitde-protein data like (protein sequence, peptide sequence, protein_ss, peptide_ss) and generate a test data file called "predict_filename"
2. Use the command `python -u preprocess_features.py predict_filename` to obtain processed feature dicts in ./preprocessing/

- STEP 2: Make predictions by the CAMP model
To predict binary peptide-protein interactions with corresponding peptide binding residues, first unzip the model file in ./model/, then use the command `python -u predict_CAMP.py predict_filename`. 


### Data curation

Here we offer the protocol to construct the benchmark dataset. Due to the copyright issues, we can not provide the complete benchmark dataset. You may follow the procedures in ./data/ to reproduce the benchmark dataset and train CAMP.

- STEP 0: Downloading data from three sources

1. RCSB PDB : Download the fasta files from ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz and pdb files
	
2. UniProt : Download pdb_chain_uniprot.tsv.gz containing the UniProt IDs of proteins in PDB complexes, which will be used to map the UniProt sequences later from https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html

3. DrugBank : Download full database.xml from DrugBank, to access the data, you need to create an account first. (requiring licenses) Then only keep DITs that belong to "Peptides" Category.

- STEP 1: Process and filter PDB data using the functions in ./data_prepare/step1_pdb_process.py

- STEP 2: Generate labels of peptide binding residues of PepPIs from PDB by the functions and detailed procedures in ./data_prepare/step2_pepBDB_pep_bindingsites.py

- STEP 3: Combine all data from DrugBank and PDB, then shuffle pairs to obtain negative samples. After that, generate sequence-based features and process the data (see STEP 0 & STEP 1 in "Predict novel pairs", you may name the data as "train_filename")

- STEP 4: use the functions in ./Cluster.py to split clusters of peptides and proteins based on sequence similarity.


