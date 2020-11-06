# -*- coding: utf-8 -*-

# Step 1:  According to the "PDB ID-Peptide Chain-Protein Chain" obtained in "step1_pdb_process.py" , retrieve the interacting information with following fields:
# ("Peptide ID","Interacting peptide residues","Peptide sequence","Interacting receptor residues","Receptor sequence(s)") a
# nd downloading the corresponding "peptide.pdb" files (please put under ./pepbdb-2020/pepbdb/$pdb_id$/peptide.pdb)

# Step 2: To map the peptide sequences from PepBDB to the peptide sequences from the peptide sequences from the RCSB PDB() generated in "step1_pdb_process.py").


# Generate query (PepBDB version) sequence file called "query_peptide.fasta" & target (RSCB PDB) fasta sequence files called "target_peptide.fasta" for peptides
# We use scripts under ./smith-waterman-src/ to align two versions of peptide sequences. The output is "alignment_result.txt"
#python query_mapping.py #to get peptide sequence vectors (the output is "peptide-mapping.txt ")
#python target_mapping.py #to get target sequence vector

# Step 3: Loading and mapping labels of binding residues for peptide sequences
# load peptide-protein pairs & pepBDB files (target : PDB fasta, query : pepBDB)
df_train = pd.read_csv('pdb_pairs', header=0, sep='#') # The output of "step1_pdb_process.py"
df_zy_pep = pd.read_csv('./pdb/peptide-mapping.txt',header=None,sep='\t')
df_zy_pep.columns= ['bdb_id','bdb_pep_seq','pep_binding_vec']
df_zy_pep['pdb_id'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[0])
df_zy_pep['pep_chain'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[1].lower())
df_zy_pep['prot_chain'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[2].upper())
df_zy_pep.drop_duplicates(['bdb_id'],inplace=True)
df_join = pd.merge(df_train, df_zy_pep, how='left', left_on=['pdb_id','pep_chain','prot_chain'],right_on=['pdb_id','pep_chain','prot_chain'])
df_v1 = df_join[['pdb_id','pep_chain','prot_chain','pep_seq','SP_PRIMARY','prot_seq','Protein_families','pep_binding_vec']]
# impute records that don't have bs information with -99999
def extract_inter_idx(pep_seq,binding_vec):
    if binding_vec==binding_vec:
        if len(binding_vec) != len(pep_seq):
            print('Error length')
        else:
            binding_lst = []
            for idx in range(len(binding_vec)):
                if binding_vec[idx]=='1':
                    binding_lst.append(idx)
        binding_str = ','.join(str(e) for e in binding_lst)
        return binding_str
    else:
        return '-99999'
    
df_v1['binding_idx'] = df_v1.apply(lambda x: extract_inter_idx(x.pep_seq,x.pep_binding_vec),axis=1)
df_part_pair = df_part_all[['pep_seq','prot_seq','binding_idx']]
df_pos_bs = pd.merge(df_v1,df_part_pair,how='left',left_on=['pep_seq','prot_seq'],right_on=['pep_seq','prot_seq'])
df_pos_bs.to_csv('pdb_pairs_bindingsites', encoding = 'utf-8', index = False, sep = ',')
