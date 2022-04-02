
# Load Intrinsic disorder
# dict : {sequence: Intrinsic Disorder Matrix}
# Intrinsic Disorder Matrix : (sequence length ,3) , last dimension :(long , short, ANCHOR score)

def extract_intrinsic_disorder(filename,ind):
    fasta_filename = filename+'.fasta'
    disorder_filename = filename+'_'+ind+'.result'
    raw_fasta_list = []
    with open('./' + fasta_filename,'r') as f:
        for line in f.readlines():
            line_list = line.strip()
            raw_fasta_list.append(line_list)

    fasta_id_list = [x for x in raw_fasta_list if x[0]=='>']
    fasta_sequence_list = [x for x in raw_fasta_list if x[0]!='>']
    fasta_seq_len_list = [len(x) for x in fasta_sequence_list]
    print(len(fasta_id_list),len(fasta_sequence_list),len(fasta_seq_len_list))

    fasta_dict = {}
    for i in range(len(fasta_id_list)):
        fasta_dict[fasta_id_list[i]]=(fasta_sequence_list[i],fasta_seq_len_list[i])

    # load protein intrinsic disorder result
    raw_result_list = []
    with open('./IntrinsicDisorder/' + disorder_filename,'r') as f:
            for line in f.readlines():
                line_list = line.strip()
                if (len(line_list)>0 and line_list[0]!='#'):
                    raw_result_list.append(line_list)

    intrinsic_id_list = [x for x in raw_result_list if x[0]=='>']
    intrinsic_score_list = [x.split('\t') for x in raw_result_list if x[0]!='>']

    start_idx = 0
    raw_score_dict = {}
    for idx in range(len(intrinsic_id_list)):
        prot_id = intrinsic_id_list[idx]
        seq_len = fasta_dict[prot_id][1]
        end_idx = start_idx + seq_len
        individual_score_list = intrinsic_score_list[start_idx:end_idx]
        individual_score_list=[x[2:] for x in individual_score_list]
        individual_score_array = np.array(individual_score_list,dtype='float')
        raw_score_dict[prot_id] = individual_score_array
        start_idx = end_idx
    print(len(fasta_dict.keys()),len(raw_score_dict.keys()))
    return fasta_dict, raw_score_dict

# long & short
fasta_dict_long, raw_score_dict_long = extract_intrinsic_disorder(fasta_filename,'long') # the input fasta file used in IUPred2A
fasta_dict_short, raw_score_dict_short = extract_intrinsic_disorder(fasta_filename,'short')

Intrinsic_score_long = {}
for key in fasta_dict_long.keys():
    sequence = fasta_dict_long[key][0]
    seq_len = fasta_dict_long[key][1]
    Intrinsic = raw_score_dict_long[key]
    if Intrinsic.shape[0]!= seq_len:
        print('Error!')
    Intrinsic_score_long[sequence] = Intrinsic

    
Intrinsic_score_short = {}
for key in fasta_dict_short.keys():
    sequence = fasta_dict_short[key][0]
    seq_len = fasta_dict_short[key][1]
    Intrinsic = raw_score_dict_short[key]
    if Intrinsic.shape[0]!= seq_len:
        print('Error!')
    Intrinsic_score_short[sequence] = Intrinsic
   


Intrinsic_score = {}
for seq in Intrinsic_score_short.keys():
    Intrinsic = Intrinsic_score_long[seq][:,0]
    short_Intrinsic = Intrinsic_score_short[seq]
    concat_Intrinsic = np.column_stack((long_Intrinsic,short_Intrinsic))
    Intrinsic_score[seq] = np.column_stack((long_Intrinsic,short_Intrinsic))


with open(output_intrisic_dict,'wb') as f: # 'output_intrisic_dict' is the name of the output dict you like
    pickle.dump(Intrinsic_score,f)

#Secondary Structure
# load predicted ss features for sequences in the dataset
def aa_ss_concat(aa,ss):
    if len(aa)!= len(ss):
        return 'string length error!'
    else:
        new_str = ''
        for i in range(len(aa)):
            concat_str = aa[i]+ss[i]+','
            new_str = new_str+concat_str
    final_str = new_str[:-1]
    return final_str


df_org = pd.read_csv('./ss/seq_data.out.ss',sep='#',header = None) #the generated file by SCRATCH1D SSPro
df_org.columns = ['col_1']


ss_idx=[]
seq_idx=[]
seq_idx = [2*x for x in list(range(int(df_org.shape[0]/2)))]
ss_idx = [x+1 for x in seq_idx]

# subset sequence dataframe and sse dataframe
df_seq = df_org.iloc[seq_idx]
df_seq.columns = ['seq_id']
df_ss = df_org.iloc[ss_idx]
df_ss.columns = ['seq_ss']

df_seq = df_seq.reset_index(drop=True)
df_ss = df_ss.reset_index(drop=True)

# join sequence & sse together
df_seq_ss = pd.merge(df_seq, df_ss,left_index=True, right_index=True)

# load id mapping file
df_id = pd.read_csv('seq_data.fasta',sep='#',header = None) #the input asta file used for SCRATCH1D SSPro
df_id.columns = ['col_1']

ss_idx=[]
seq_idx=[]
seq_idx = [2*x for x in list(range(int(df_id.shape[0]/2)))]
ss_idx = [x+1 for x in seq_idx]

# subset sequence dataframe and sse dataframe
df_seq = df_id.iloc[seq_idx]
df_seq.columns = ['seq_id']
df_ss = df_id.iloc[ss_idx]
df_ss.columns = ['seq']

df_seq = df_seq.reset_index(drop=True)
df_ss = df_ss.reset_index(drop=True)

# join sequence &  sse together
df_idx = pd.merge(df_seq, df_ss,left_index=True, right_index=True)
df_output_ss = pd.merge(df_idx, df_seq_ss, left_on=['seq_id'], right_on=['seq_id'])
df_output_ss['concat_seq'] = df_output_ss.apply(lambda x: aa_ss_concat(x['seq'],x['seq_ss']),axis=1)
df_output_ss.to_csv(output_ss_filename, encoding = 'utf-8', index = False, sep = '\t') # 'output_ss_filename' is the name of the output tsv you like

### Load Protein PSSM Files (first change the value of protein_number)
# prot_pssm_dict : key is protein sequence, value is protein PSSM Matrix
prot_pssm_dict_all={}
prot_pssm_dict={}
protein_num = 0 ### NEED TO BE CHANGED TO the total number of protein sequences
for i in range(protein_num):
    filename_pssm = 'new_prot_'+str(i)+'.pssm' # need to name each individual fasta and pssm file with the same prefix
    filename_fasta = 'new_prot_'+str(i)+'.fasta'
    prot_key = 'new_prot_'+str(i)
    pssm_line_list= []
    
    with open('./pssm/prot_file/'+filename_fasta,'r') as f: # directory to store fasta files (single file of each protein)
        for line in f.readlines():
            prot_seq = line.strip()
    
    with open('./pssm/pssm_result/'+filename_pssm,'r') as f:  # directory to store pssm files (single file of each protein)
        for line in f.readlines()[3:-6]:
            line_list = line.strip().split(' ')
            line_list = [x for x in line_list if x!=''][2:22]
            line_list = [int(x) for x in line_list]
            if len(line_list)!=20:
                print('Error line:')
                print(line_list)
            pssm_line_list.append(line_list)
        pssm_array = np.array(pssm_line_list)
        if pssm_array.shape[1]!=20:
            print('Error!')
            print(filename_pssm)
        else:
            prot_pssm_dict_all[prot_key] = (prot_seq,pssm_array)
            prot_pssm_dict[prot_seq]=pssm_array

with open(output_pssm_dict,'wb') as f:  # 'output_pssm_dict' is the name of the output dict you like
    pickle.dump(prot_pssm_dict,f)
            
