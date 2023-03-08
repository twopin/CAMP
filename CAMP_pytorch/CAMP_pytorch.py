from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle 
import math
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, classification_report,average_precision_score
import pandas as pd
import torch.optim as optim
import os
import torchsnooper
#os.environ['CUDA_VISIBLE_DEVICES'] ='1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

def cls_scores(label, pred):
	label = label.reshape(-1)
	pred = pred.reshape(-1)
	# r2_score, mean_squred_error are ignored
	return roc_auc_score(label, pred), average_precision_score(label, pred)

task='cls'
name = 'peptide'
with open('./preprocessing_v2/'+task+'_'+name+'_protein_feature_dict','rb') as f: 
    prot_seq_dict = pickle.load(f,encoding="latin1")
with open('./preprocessing_v2/'+task+'_'+name+'_peptide_feature_dict','rb') as f: 
    pep_seq_dict = pickle.load(f,encoding="latin1")
with open('./preprocessing_v2/'+task+'_'+name+'_protein_ss_feature_dict','rb') as f: 
	prot_ss_feature_dict = pickle.load(f,encoding="latin1")
with open('./preprocessing_v2/'+task+'_'+name+'_compound_ss_feature_dict','rb') as f:
    pep_ss_feature_dict = pickle.load(f,encoding="latin1")
with open('./preprocessing_v2/'+task+'_'+name+'_protein_dense_feature_dict','rb') as f: 
	prot_dense_feature_dict = pickle.load(f,encoding="latin1")
with open('./preprocessing_v2/'+task+'_'+name+'_compound_dense_feature_dict','rb') as f:
    pep_dense_feature_dict = pickle.load(f,encoding="latin1")
with open('./preprocessing_v2/'+task+'_'+name+'_protein_2_feature_dict','rb') as f: 
    prot_2_feature_dict = pickle.load(f,encoding="latin1")
with open('./preprocessing_v2/'+task+'_'+name+'_compound_2_feature_dict','rb') as f:
    pep_2_feature_dict = pickle.load(f,encoding="latin1")


print('load feature dict')
X_pep, X_prot, X_pep_SS, X_prot_SS, X_pep_2, X_prot_2 = [], [], [], [], [], []
X_dense_pep,X_dense_prot = [],[]
pep_sequence, prot_sequence, Y = [], [], []
with open('train_ss_v2_0304_5_v2') as f:
	for line in f.readlines()[1:]:
		protein, peptide, label, pep_ss, prot_ss  = line.strip().split('\t')
		pep_sequence.append(peptide)
		prot_sequence.append(protein)
		Y.append(label)
		X_pep.append(pep_seq_dict[peptide])
		X_prot.append(prot_seq_dict[protein])
		X_pep_SS.append(pep_ss_feature_dict[pep_ss])
		X_prot_SS.append(prot_ss_feature_dict[prot_ss])
		X_pep_2.append(pep_2_feature_dict[peptide])
		X_prot_2.append(prot_2_feature_dict[protein])
		X_dense_pep.append(pep_dense_feature_dict[peptide])
		X_dense_prot.append(prot_dense_feature_dict[protein])
		
X_pep = np.array(X_pep)
X_prot = np.array(X_prot)
X_pep_SS = np.array(X_pep_SS)
X_prot_SS = np.array(X_prot_SS)
X_pep_2 = np.array(X_pep_2)
X_prot_2 = np.array(X_prot_2)
X_dense_pep = np.array(X_dense_pep)
X_dense_prot = np.array(X_dense_prot)
Y = np.array(Y).astype(int)
pep_sequence = np.array(pep_sequence)
prot_sequence = np.array(prot_sequence)
sf_idx = list(range(Y.shape[0]))
np.random.shuffle(sf_idx)
X_pep = X_pep[sf_idx]
X_prot = X_prot[sf_idx]
X_pep_SS = X_pep_SS[sf_idx]
X_prot_SS = X_prot_SS[sf_idx]
X_pep_2 = X_pep_2[sf_idx]
X_prot_2 = X_prot_2[sf_idx]
X_dense_pep = X_dense_pep[sf_idx]
X_dense_prot = X_dense_prot[sf_idx]
Y = Y[sf_idx]
pep_sequence = pep_sequence[sf_idx]
prot_sequence = prot_sequence[sf_idx]


def random_split(X, Y, fold=5):
	skf = StratifiedKFold(n_splits=fold, shuffle=True)
	train_idx_list, test_idx_list = [], []
	for train_index, test_index in skf.split(X, Y):
		train_idx_list.append(train_index)
		test_idx_list.append(test_index)
	return train_idx_list, test_idx_list
train_idx_list, test_idx_list = random_split(X_pep, Y, fold=5)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
    def forward(self,x):
        output, _ = torch.max(x,1)
        return output

class ConvNN(nn.Module):
    def __init__(self,in_dim,c_dim,kernel_size):
        super(ConvNN,self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels= c_dim, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim, out_channels= c_dim*2, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim*2, out_channels= c_dim*3, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            #GlobalMaxPool1d() # 192
            )
    def forward(self,x):
        x = self.convs(x)
        return x

class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / math.sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output


class CAMP(nn.Module):
    def __init__(self):
        super(CAMP,self).__init__()
        #self.config = config
        self.embed_seq = nn.Embedding(65+1, 128) # padding_idx=0, vocab_size = 65/25, embedding_size=128
        self.embed_ss = nn.Embedding(75+1,128)
        self.embed_two = nn.Embedding(7+1,128)
        self.pep_convs = ConvNN(512,64,7)
        self.prot_convs = ConvNN(512,64,8)
        self.pep_fc = nn.Linear(3,128)    
        self.prot_fc = nn.Linear(23,128)
        self.global_max_pooling = GlobalMaxPool1d()
        #self.dnns = DNN(config.in_dim,config.d_dim1,config.d_dim2,config.dropout)
        self.dnns = nn.Sequential(
            nn.Linear(640,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,512))
        
        self.att = Self_Attention(128,128,128)
        #c_dim
        self.output = nn.Linear(512,1)

    #@torchsnooper.snoop()
    def forward(self, x_pep,x_prot,x_pep_ss,x_prot_ss,x_pep_2,x_prot_2,x_pep_dense,x_prot_dense):

        pep_seq_emb = self.embed_seq(x_pep.long())#.type(torch.LongTensor))
        prot_seq_emb = self.embed_seq(x_prot.long())#.type(torch.LongTensor))
        pep_ss_emb = self.embed_ss(x_pep_ss.long())#type(torch.LongTensor))
        prot_ss_emb = self.embed_ss(x_prot_ss.long())
        pep_2_emb = self.embed_two(x_pep_2.long())
        prot_2_emb = self.embed_two(x_prot_2.long())
        pep_dense = self.pep_fc(x_pep_dense)
        prot_dense = self.prot_fc(x_prot_dense)
        

        encode_peptide = torch.cat([pep_seq_emb, pep_ss_emb, pep_2_emb, pep_dense],dim=-1)
        encode_protein = torch.cat([prot_seq_emb, prot_ss_emb, prot_2_emb, prot_dense],dim=-1)

        encode_peptide = encode_peptide.permute(0,2,1)
        encode_protein = encode_protein.permute(0,2,1)

        encode_peptide = self.pep_convs(encode_peptide)
        encode_peptide = encode_peptide.permute(0,2,1)
        encode_peptide_global = self.global_max_pooling(encode_peptide)

        encode_protein = self.prot_convs(encode_protein)
        encode_protein = encode_protein.permute(0,2,1)
        encode_protein_global = self.global_max_pooling(encode_protein)
        
        # self-attention
        pep_seq_att = self.embed_seq(x_pep.long())
        peptide_att = self.att(pep_seq_att)
        peptide_att = self.global_max_pooling(peptide_att)
        
        prot_seq_att = self.embed_seq(x_prot.long())
        protein_att = self.att(prot_seq_att)
        protein_att = self.global_max_pooling(protein_att)

        encode_interaction = torch.cat([encode_peptide_global,encode_protein_global,peptide_att,protein_att],axis=-1)
        encode_interaction = self.dnns(encode_interaction)
        predictions = torch.sigmoid(self.output(encode_interaction))

        return predictions.squeeze(dim=1)

ft_idx_lst = [50,850,900,1700,1750,2550,2700,21100]
#@torchsnooper.snoop()
def train(model,dataloader,device,criterion,optimizer):
    model.train()
    preds,labels = [], []
    avg_loss = 0
    criterion.to(device)
    #print('start loading')
    for batch, (X_load,Y_load) in enumerate(dataloader):
        X = X_load.to(device)
        Y = Y_load.to(device)
        X_pep = X[:,:ft_idx_lst[0]].to(device)
        X_prot = X[:,ft_idx_lst[0]:ft_idx_lst[1]].to(device)
        X_pep_ss = X[:,ft_idx_lst[1]:ft_idx_lst[2]].to(device)
        X_prot_ss = X[:,ft_idx_lst[2]:ft_idx_lst[3]].to(device)
        X_pep_2 = X[:,ft_idx_lst[3]:ft_idx_lst[4]].to(device)
        X_prot_2 = X[:,ft_idx_lst[4]:ft_idx_lst[5]].to(device)
        X_pep_dense = X[:,ft_idx_lst[5]:ft_idx_lst[6]].reshape(X.shape[0],50,3).to(device)
        X_prot_dense = X[:,ft_idx_lst[6]:ft_idx_lst[7]].reshape(X.shape[0],800,23).to(device)

        optimizer.zero_grad()
        #print(X_pep.is_cuda,X_prot.is_cuda,X_pep_ss.is_cuda,X_prot_ss.is_cuda,X_pep_2.is_cuda,X_prot_2.is_cuda,X_pep_dense.is_cuda,X_prot_dense.is_cuda,Y.is_cuda)
        pred=model(X_pep,X_prot,X_pep_ss,X_prot_ss,X_pep_2,X_prot_2,X_pep_dense,X_prot_dense)
        preds.extend(pred.detach().cpu().numpy().tolist())
        labels.extend(Y.detach().cpu().numpy().tolist())

        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    preds = np.array(preds)
    labels = np.array(labels)

    acc = accuracy_score(labels, np.round(preds))
    test_scores = cls_scores(labels,preds)
    AUC = round(test_scores[0],6)
    AUPR = round(test_scores[1],6)
    avg_loss /= len(dataloader)

    return avg_loss, AUC, AUPR

def test(model,dataloader,device,criterion):
    model.eval()
    preds,labels = [], []
    with torch.no_grad():
        avg_loss = 0
        for batch, (X,Y) in enumerate(dataloader):
            X = X.to(device)
            Y = Y.to(device)
            X_pep = X[:,:ft_idx_lst[0]]
            X_prot = X[:,ft_idx_lst[0]:ft_idx_lst[1]]
            X_pep_ss = X[:,ft_idx_lst[1]:ft_idx_lst[2]]
            X_prot_ss = X[:,ft_idx_lst[2]:ft_idx_lst[3]]
            X_pep_2 = X[:,ft_idx_lst[3]:ft_idx_lst[4]]
            X_prot_2 = X[:,ft_idx_lst[4]:ft_idx_lst[5]]
            X_pep_dense = X[:,ft_idx_lst[5]:ft_idx_lst[6]].reshape(X.shape[0],50,3)
            X_prot_dense = X[:,ft_idx_lst[6]:ft_idx_lst[7]].reshape(X.shape[0],800,23)

            pred=model(X_pep,X_prot,X_pep_ss,X_prot_ss,X_pep_2,X_prot_2,X_pep_dense,X_prot_dense)
            preds.extend(pred.detach().cpu().numpy().tolist())
            labels.extend(Y.detach().cpu().numpy().tolist())

            loss = criterion(pred, Y)
            avg_loss += loss.item()

        avg_loss /= len(dataloader)

    preds = np.array(preds)
    labels = np.array(labels)

    acc = accuracy_score(labels, np.round(preds))
    test_scores = cls_scores(labels,preds)
    AUC = round(test_scores[0],6)
    AUPR = round(test_scores[1],6)
    
    return avg_loss, AUC, AUPR

def load_checkpoint(filepath):
    ckpt = torch.load(filepath)
    model = ckpt['model']
    model.load_state_dict(ckpt['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad=False
    model.eval()
    return model

n_fold = 5
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
test_AUC_list, test_AUPR_list = [],[]
for fold in range(n_fold):
    train_idx,test_idx = train_idx_list[fold],test_idx_list[fold]
    print(f"Train:{len(train_idx)} and test: {len(test_idx)} of fold: {fold};")

    train_X_dense_pep = X_dense_pep[train_idx].reshape(X_dense_pep[train_idx].shape[0],-1)
    train_X_dense_prot = X_dense_prot[train_idx].reshape(X_dense_prot[train_idx].shape[0],-1)

    test_X_dense_pep = X_dense_pep[test_idx].reshape(X_dense_pep[test_idx].shape[0],-1)
    test_X_dense_prot = X_dense_prot[test_idx].reshape(X_dense_prot[test_idx].shape[0],-1)


    train_X = np.concatenate([X_pep[train_idx],X_prot[train_idx],
                X_pep_SS[train_idx],X_prot_SS[train_idx],
                X_pep_2[train_idx],X_prot_2[train_idx],
                train_X_dense_pep,train_X_dense_prot,
                ],axis=1)
    test_X = np.concatenate([X_pep[test_idx],X_prot[test_idx],
                X_pep_SS[test_idx],X_prot_SS[test_idx],
                X_pep_2[test_idx],X_prot_2[test_idx],
                test_X_dense_pep,test_X_dense_prot,
                ],axis=1)
    train_Y,test_Y = np.array(Y[train_idx]),np.array(Y[test_idx])

    train_sequence = [prot_sequence[train_idx],pep_sequence[train_idx]]
    test_sequence = [prot_sequence[test_idx],pep_sequence[test_idx]]

    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_Y = torch.from_numpy(train_Y)
    test_Y = torch.from_numpy(test_Y)

    train_X = train_X.float()
    train_Y = train_Y.float()
    test_X = test_X.float()
    test_Y = test_Y.float()

    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    train_dataset = TensorDataset(train_X, train_Y)
    test_dataset = TensorDataset(test_X, test_Y)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

    print("Start training and device in use:",device)
    EPOCHS = 100
    model = CAMP()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = optim.RMSprop(model.parameters(),lr=0.0005) #weight_decay=1e-6
    for e in range(EPOCHS):
        train_loss, train_AUC, train_AUPR = train(model, train_loader, device, criterion, optimizer)
        test_loss, test_AUC, test_AUPR = test(model, test_loader, device, criterion)
        print(f"Epoch:{e+1};")

        #print(f"Training loss:{train_loss:.4f}, ACC:{train_acc:.4f}, AUC:{train_auc:.4f}, cls AUC:{train_AUC:.6f}, cls AUPR:{train_AUPR:.6f}")
        #print(f"Test loss:{test_loss:.4f}, ACC:{test_acc:.4f}, AUC:{test_auc:.4f}, cls AUC:{test_AUC:.6f}, cls AUPR:{test_AUPR:.6f}")
        print(f"Training loss:{train_loss:.4f}, AUC:{train_AUC:.6f}, AUPR:{train_AUPR:.6f}")
        print(f"Test loss:{test_loss:.4f}, AUC:{test_AUC:.6f}, AUPR:{test_AUPR:.6f}")
    test_AUC_list.append(test_AUC)
    test_AUPR_list.append(test_AUPR)

    print(f'Save model of epoch {e} with {n_fold}-fold cv')
    checkpoint = {'model':CAMP(),'model_state_dict':model.state_dict()}
    torch.save(checkpoint,f'./ckpts/model_cv_ckpts_{fold}.pkl')

#     model_ckpt = load_checkpoint(f'./ckpts/model_full_ckpts_{fold}.pkl')
#     model_ckpt=model_ckpt.to(device)
#     test_loss_ckpt, test_AUC_ckpt, test_AUPR_ckpt = test(model_ckpt, test_loader, device, criterion)
#     print(f"Test loss:{test_loss_ckpt:.4f}, AUC:{test_AUC_ckpt:.6f}, AUPR:{test_AUPR_ckpt:.6f}")


print('fold mean auc & aupr', np.mean(test_AUC_list, axis=0),np.mean(test_AUPR_list, axis=0))
print('fold std auc & aupr', np.std(test_AUC_list, axis=0),np.std(test_AUPR_list, axis=0))

        


            
