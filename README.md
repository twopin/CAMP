# CAMP: a Convolutional Attention-based Neural Network for Multi-level Peptide-protein Interaction Prediction

CAMP is a sequence-based deep learning framework for multifaceted prediction of peptide-protein interactions, including not only binary peptide-protein interactions, but also corresponding peptide binding residues.

### UFRC Notes
#### 1. navigate to directory 
```
cd C:\Users\djlemas\OneDrive\Documents\CAMP
```
#### 2. pull docker image
```
docker pull tensorflow/tensorflow:1.2.1-gpu 
```

#### 3. boot into image 
```
docker run --rm -it -v ${PWD}:/home/camp tensorflow/tensorflow:1.2.1-gpu bash
```

#### 4. #> navigate to /camp 
```
#> cd /home/camp
```

#### 5. #> run environment.yml to create camp_env 
```
conda env create -f environment.yml

```
#### 6. #> activate camp_env 
```
conda activate camp_env

```

### Notice

Since CAMP exploits a series of sequence-based features, you CANNOT use CAMP when the primary sequence information of peptides and proteins is unknown. In addition, you need to replace any non-standard amino acid with 'X' first. Pay attention to that if you want to train with your own data.

### Requirement

Trained and tested on a linux server with GeForce GTX 1080 and the running environment is as follows:
Python2.7, Keras=2.0.8, Tensorflow=1.2.1, RDKit (`conda install -y -c conda-forge rdkit`), CUDA (GPU computation)

### Running CAMP

1. To predict with our example code, first create a directory `example_prediction` and unzip the feature dictionaries by `unzip example_data_feature.zip` and use the test data with the command `python -u predict_CAMP.py`.
Options are:  
`-b: The batch size, default: 256.`  
`-l: The padding length for peptides, default: 50.`  
`-f: The fraction of GPU memory to process the model, default: 0.9.`  
`-m: The prediction model, 1 indicates binary prediction task and 2 indicates dual predictions, default: 1.`  
`-p: The The padding length for proteins p, default: 800.`  
2. To generate features, please refer to the instructuion in "Data curation".
3. We recently added `run.sh` for users with no ML backgrounds to easiley test the model. Just simply run command `sh run.sh`.

### Data curation

Here we offer the protocol to construct the benchmark dataset. Due to the copyright issues, we can not provide the complete benchmark dataset. You may follow the procedures in ./data_prepare/ to reproduce the benchmark dataset and corresponding features (or features of your own data). For cluster-based cross validations, after finishing all the following procedures, you can get alignment results from https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library and use the scripts in ./cluster/ to obtain the similarity matrix and split clusters of peptides and proteins based on the similarity matrix.

- STEP 0: Downloading data from three sources

1. RCSB PDB : Download the fasta files from ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz and pdb files
	
2. UniProt : Download pdb_chain_uniprot.tsv.gz containing the UniProt IDs of proteins in PDB complexes, which will be used to map the UniProt sequences later from https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html

3. DrugBank : Download full database.xml from DrugBank, to access the data, you need to create an account first. (requiring licenses) Then only keep DITs that belong to "Peptides" Category.

- STEP 1: Process and filter PDB data using the functions in ./data_prepare/step1_pdb_process.py

- STEP 2: Generate labels of peptide binding residues of PepPIs from PDB by the functions and detailed procedures in ./data_prepare/step2_pepBDB_pep_bindingsites.py

- STEP 3: Generated sequnece-based features
Combine all data from DrugBank and PDB, then shuffle pairs to obtain negative samples. After that, you need to generate the corresponding features (you can either use the online servers or download the softwares):

	1. Secondary structure : http://scratch.proteomics.ics.uci.edu/explanation.html#SSpro
	2. Intrinsic Disorder :  https://iupred2a.elte.hu/
	3. PSSM matrix: ftp://ftp.cnbi.nlm.nih.gov/blast/executables/blast+/LATEST/
	4. Use the functions in ./data_prepare/step3_generate_features.py to process raw output files. Then put all feature dicts in ./dense_feature_dict/

- STEP 4: Proprecess data with feature dictionaries

1. format the peptide-protein data like (protein sequence, peptide sequence, protein_ss, peptide_ss) and generate a test data file called "test_filename" for 
2. Use the command `python -u preprocess_features.py test_filename` to obtain processed feature dicts in ./preprocessing/

### Recent Update 07/02/2021

Fixed bugs and uploaded run.sh to easily test the model.

### License

This software is copyrighted by Machine Learning and Computational Biology Group @ Tsinghua University.

The algorithm and data can be used only for NON COMMERCIAL purposes.
