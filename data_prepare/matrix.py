# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:59:48 2020

@author: lenovo
"""
import numpy as np
import csv
query_pep_sequ_index={}
query_prot_sequ_index={}
with open('query_peptide_sequence_index.txt','r') as f1:
    for line in f1:
        line=line.split('\t')
        query_id=line[0]
        sequ_index=line[1].strip('\n')
        query_pep_sequ_index[query_id]=sequ_index
with open('query_prot_sequence_index.txt','r') as f2:
    for line in f2:
        line=line.split('\t')
        query_id=line[0]
        sequ_index=line[1].strip('\n')
        query_prot_sequ_index[query_id]=sequ_index
pep_pdb_uniprot_map={}
prot_pdb_uniprot_map={}
with open('pep_query_target_mapping.txt','r') as f3:
    for line in f3:
        line=line.split('\t')
        pdb_id=line[0]
        mapping=line[1].strip('\n')
        pep_pdb_uniprot_map[pdb_id]=mapping
with open('prot_query_target_mapping.txt','r') as f4:
    for line in f4:
        line=line.split('\t')
        pdb_id=line[0]
        mapping=line[1].strip('\n')
        prot_pdb_uniprot_map[pdb_id]=mapping
peptide_sequence={}
prot_sequence={}
with open('target_peptide.fasta') as f5:
    for line in f5:
        if line.startswith('>'):
            qid=line[1:].strip('\n')
        else:
            sequ=line.strip('\n')
            peptide_sequence[qid]=sequ
with open('target_prot.fasta') as f6:
    for line in f6:
        if line.startswith('>'):
            qid=line[1:].strip('\n')
        else:
            sequ=line.strip('\n')
            prot_sequence[qid]=sequ
pdbids=[]
csvfile=open('crawl_results.csv','r')
reader=csv.reader(csvfile)
for item in reader:
    if reader.line_num==1:
        continue
    pdbids.append(item[0])
problem_pdbid=[]
for pdbid in pdbids:
    if pdbid not in prot_pdb_uniprot_map or pdbid not in pep_pdb_uniprot_map:
        continue
    if pdbid not in query_pep_sequ_index or pdbid not in query_prot_sequ_index:
        continue
    pep_index=query_pep_sequ_index[pdbid]
    pep_index=pep_index.split(',')
    pep_index.pop()
    prot_index=query_prot_sequ_index[pdbid]
    prot_index=prot_index.split(',')
    prot_index.pop()
    peptide_chain=pdbid[5]
    pep_binding=[]
    prot_chain=pdbid[7]
    prot_binding=[]
    original_pep_prot_binding=[]
    with open('inter_data/'+pdbid[:6]+'.dat','r') as f:
        for line in f:
            line=line.split()
            if line[0]==prot_chain and line[2]==peptide_chain:
                prot=line[1]
                prot_binding.append(prot)
                pep=line[3]
                pep_binding.append(pep)
                original_pep_prot_binding.append(list([prot,pep]))
    #mapping binding data to query
    query_pep_prot_binding=[]
    for item in original_pep_prot_binding:
        prot=item[0]
        pep=item[1]
        if prot not in prot_index:
            problem_pdbid.append(pdbid)
            continue
            prot=prot+'A'
        if pep not in pep_index:
            problem_pdbid.append(pdbid)
            continue
            pep=pep+'A'
        new_prot=prot_index.index(prot)
        new_pep=pep_index.index(pep)
        query_pep_prot_binding.append(list([new_prot,new_pep]))
    #mapping query to target
    target_pep_prot_binding=[]
    pep_target_query_mapping=pep_pdb_uniprot_map[pdbid]
    pep_target_query_mapping=pep_target_query_mapping.split(',')
    pep_target_query_mapping.pop()
    prot_target_query_mapping=prot_pdb_uniprot_map[pdbid]
    prot_target_query_mapping=prot_target_query_mapping.split(',')
    prot_target_query_mapping.pop()
    prot_target_len=len(prot_sequence[pdbid])
    pep_target_len=len(peptide_sequence[pdbid])
    binding_array=np.zeros((prot_target_len,pep_target_len),dtype=int)
    for item in query_pep_prot_binding:
        prot=item[0]
        pep=item[1]
        if str(prot) not in prot_target_query_mapping:
            continue
        final_prot=prot_target_query_mapping.index(str(prot))
        if str(pep) not in pep_target_query_mapping:
            continue
        final_pep=pep_target_query_mapping.index(str(pep))
        target_pep_prot_binding.append(list([final_prot,final_pep]))
    for item in target_pep_prot_binding:
        line=item[0]
        col=item[1]
        binding_array[line][col]=1
    np.savetxt('target_metrix/'+pdbid+'.csv', binding_array, fmt="%d",delimiter = ',') 
with open('problem.txt','w') as f:
    new=[]
    for qid in problem_pdbid:
        if qid not in new:
            new.append(qid)
            f.write(qid+'\n')
print('down')
