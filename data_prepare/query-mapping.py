# -*- coding: utf-8 -*-

querys=[]
import csv
import os
def check_abnormal_aa(peptide_seq):
    len_seq = len(peptide_seq)
    cnt = 0
    standard_aa = ['G','A','P','V','L','I','M','F','Y','W','S','T','C','N','Q','K','H','R','D','E']
    for i in peptide_seq:
        if i in standard_aa :
            cnt = cnt+1
    score = float(cnt)/len_seq
    return score
def delete_duplicate(seq):
    seqsort=[]
    for i in seq:
        if i not in seqsort:
            seqsort.append(i)
    seqstr={}
    for s in seqsort:
        seqstr[s[:-1]]=s[-1:]    
    return seqstr
aa_dict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', \
           'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', \
           'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'SEC': 'U', 'PLY': 'O'}
csvfile=open('crawl_results.csv','r')
reader=csv.reader(csvfile)
residue_dict={}
seq_dict={}
    
for item in reader:
    if reader.line_num==1:
        continue
    qid=item[0]
    querys.append(qid)
    pep_index=item[1].split(': ') #prot_index=item[3].split(': ')
    residue_dict[item[0]]=pep_index[1]
    seq_dict[item[0]]=item[2].split(': ')[1]#seq_dict[item[0]]=item[4].split(': ')[1]
whole_dict={}
for pid in querys:
        sequence=[]
    
        pdbid=pid[:6]
        chain=pid[5] #chain=pid[7]
        address='pepbdb-20200318/pepbdb/'+pdbid+'/peptide.pdb'#receptor.pdb
        if os.path.isfile(address)==False:
            continue
        with open(address,'r') as f:
            for line in f:
                line=line.split()
                if 'HETATM' in line[0] and len(line[0])>6:
                    if line[3]==chain:
                            index=line[4]
                    elif line[3][0]==chain:
                            index=line[3][1:]
                    else:
                            continue
                    amino = line[2]
                    if amino in aa_dict:
                            amino=aa_dict[amino]
                    else:
                            amino='X'
                    sequence.append(index+amino)
                else:
                    if line[0]=='TER':
                            continue
                    amino=line[3]
                    if line[4]==chain:
                        index=line[5]
                    elif line[4][0]==chain:
                        index=line[4][1:]
                    else:
                        continue
                    if amino in aa_dict:
                            amino=aa_dict[amino]
                    else:
                            amino='X'
                    sequence.append(index+amino)
            stringdict=delete_duplicate(sequence)
        whole_dict[pid]=stringdict
starts={}
sorted_lists={}
pass_lists=[]
with open('peptide-mapping.txt','w') as f2:#prot-mapping.txt
    for pdbid in querys:
        if pdbid not in residue_dict:
            print(pdbid)
            continue
        residue=residue_dict[pdbid].split(', ')
        output=''
        outputseq=''
        real_sequ=whole_dict[pdbid]
        query_sequ=seq_dict[pdbid]
        for i in real_sequ:
            output+=real_sequ[i]
        index=output.find(query_sequ)
        if index==-1:
            continue
            #print(pdbid)
        flag=0
        for i in real_sequ:
            if flag==index:
                start=i
                #print(i)
                break
            else:
                flag+=1
        new_dict={}
        sorted_list=list(real_sequ.keys())
        sorted_list=sorted_list[flag:flag+len(query_sequ)]
        sorted_lists[pdbid]=sorted_list
        for i in sorted_list:
            new_dict[i]=real_sequ[i]
        for i in new_dict:
            if i in residue:
                outputseq+='1'
            else:
                outputseq+='0'
        f2.write(pdbid+'\t'+query_sequ+'\t'+outputseq+'\n')
with open('query_peptide_sequence_index.txt') as f3:      
        for pdbid in sorted_lists:
            sequ_index=''
            sorted_list=sorted_lists[pdbid]
            for i in sorted_list:
                sequ_index+=i+','
            f3.write(pdbid+'\t'+sequ_index)
        