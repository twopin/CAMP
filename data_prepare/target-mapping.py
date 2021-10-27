# -*- coding: utf-8 -*-

import csv
def get_result_dict():
    result_dict = {}
    f = open('alignment_result.txt')
    i = -1
    seq_target, seq_query, align = '', '', ''
    pdb_ratio_dict = {}
    for line in f.readlines():
        i += 1
        if i%4 == 0:
            if 'target_name' in line:
                if len(seq_target) != 0:
                    result_dict[target_name] = (seq_target, seq_query, align, target_start, query_start)
                target_name = line.strip().split(' ')[-1]
                #print('target_name',target_name)
                seq_target, seq_query, align = '', '', ''
            else:
                seq_target += line.split('\t')[1]
                #print('seq_target',seq_target)
        elif i%4 == 1:
            if 'query_name' in line:
                query_name = line.strip().split(' ')[-1]
                #print('query_name',query_name)
            else:
                align += line.strip('\n').split('\t')[1]
                #print('align',align)
        elif i%4 == 2:
            if 'optimal_alignment_score' in line:
                for item in line.strip().split('\t'):
                    if item.split(' ')[0] == 'target_begin:':
                        target_start = int(item.split(' ')[1])
                    elif item.split(' ')[0] == 'query_begin:':
                        query_start = int(item.split(' ')[1])
            else:
                seq_query += line.split('\t')[1]
    f.close()
    return result_dict
def seq_with_gap_to_idx(seq):
    idx_list = []
    i = 0
    for aa in seq:
        if aa == '-':
            idx_list.append(-1)
        else:
            idx_list.append(i)
            i += 1
    return idx_list

def get_target_idx(target_idx_list, query_idx_list, align, target_start, query_start):
    pdb_to_uniprot_idx = []
    for i in range(target_start-1):
        pdb_to_uniprot_idx.append(-1)
    for i in range(len(target_idx_list)):
        if target_idx_list[i] != -1:
            if align[i]  == '|' and query_idx_list[i] != -1:
                pdb_to_uniprot_idx.append(query_idx_list[i] + query_start-1)
            else:
                pdb_to_uniprot_idx.append(-1)
    return pdb_to_uniprot_idx

def get_pdb_to_uniprot_map(result_dict):
    #pdb_ratio_dict = {}
    pdb_to_uniprot_map_dict = {}
    for name in result_dict:
        #pdbid, chain, uniport_id = name.split('_')
        record_id = name
        seq_target, seq_query, align, target_start, query_start = result_dict[name]
        ratio = float(align.count('|'))/float(len(seq_target.replace('-','')))
        if ratio < 0.2:
            continue

        target_idx_list = seq_with_gap_to_idx(seq_target)
        query_idx_list = seq_with_gap_to_idx(seq_query)
        pdb_to_uniprot_idx = get_target_idx(target_idx_list, query_idx_list, align, target_start, query_start)
        if record_id in pdb_to_uniprot_map_dict:
            pdb_to_uniprot_map_dict[record_id] = pdb_to_uniprot_idx
        else:
            pdb_to_uniprot_map_dict[record_id] = {}
            pdb_to_uniprot_map_dict[record_id] = pdb_to_uniprot_idx
    return pdb_to_uniprot_map_dict
def get_query_residue():
    residues={}
    with open('peptide-mapping.txt','r') as f3:
        for line in f3:
            line=line.split('\t')
            qid=line[0]
            if len(line)!=3:
                continue
            vector=line[2]
            index=0
            residue=[]
            for i in vector:
                if i=='1':
                    residue.append(index)
                    index+=1
                else:
                    index+=1
            residues[qid]=residue
    return residues
def get_target_sequence():
    target={}
    with open('target-peptide.fasta','r') as f2: 
        for line in f2.readlines():
            if line.startswith('>'):
                name=line[1:9]
            else:
                seq=line.strip('\n')
                target[name]=seq
    return target
peptide_resolution_dict=get_result_dict()
pdb_uniprot_map=get_pdb_to_uniprot_map(peptide_resolution_dict)
query_residues=get_query_residue()
target_sequence=get_target_sequence()
with open('peptide-mapping.txt','w') as f1:
    for item in pdb_uniprot_map:
        outputseq=''
        pdb_to_uniprot_idx=pdb_uniprot_map[item]
        sequence=target_sequence[item]
        #sequence_index=sequence_indexs[item]
        if item not in query_residues:
            continue
        query_residue=query_residues[item]
        if len(sequence)>len(pdb_to_uniprot_idx):
            last=pdb_to_uniprot_idx[len(pdb_to_uniprot_idx)-1]
            for i in range(len(sequence)-len(pdb_to_uniprot_idx)):
                pdb_to_uniprot_idx.append(-1)
        for idx in pdb_to_uniprot_idx:
            if idx in query_residue:
                outputseq+='1'
            else:
                outputseq+='0'
        f1.write(item+'\t'+sequence+'\t'+outputseq+'\n')
            
        
    
    