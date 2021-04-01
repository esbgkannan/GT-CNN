from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.utils.data import Dataset, DataLoader

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import RandomOverSampler

from collections import OrderedDict
import json
import subprocess
import sys
import time
import xml.etree.ElementTree

import os
import io
import random

from scipy.stats import norm
from scipy.stats import genextreme

######################################################################

# Functions for preproccsing

#### This function is intended to concatenate seq processed by NetSurfP into a single row, needed when new seqs are provided
# a helper function to concatenate seq into a row

def Transfer_Function(Data, val = False, fold_name=False, family_name=False):
    NameList = Data['id'].unique()
    ReturnData = pd.DataFrame(columns = ["Name", "fold", "family", "q3seq", "rawseq"])
    len_sequences = []
    for _ in range(len(NameList)):
        seq = Data[Data['id'].isin([NameList[_]])]
        q3seq = ''
        rawSeq = ''
        if val == True:
            Fold = fold_name
            Fam = family_name
        else:
            Fold = (NameList[_].split("|")[0]).split("-")[1]
            Fam = (NameList[_].split("|")[0])
        for row in seq.itertuples():
            q3seq += str(getattr(row, 'q3'))
            rawSeq += str(getattr(row, 'seq'))
        Name = NameList[_]
        len_sequences.append(len(q3seq))
        ReturnData = ReturnData.append(pd.DataFrame([[Name, Fold, Fam, q3seq, rawSeq]], columns = ["Name","fold","family", "q3seq", "rawseq"]), ignore_index=True)
    return pd.DataFrame(ReturnData), pd.DataFrame(len_sequences)

#### The following 2 function are intended to cut the sequence based on domain bound.

## Add domain bound information to the seq file by matching the IDs
def Add_Domain_Bound(Data, Domain_bound_file):
    returnDate = pd.DataFrame(columns=Data.columns)
    df_seq = pd.DataFrame(columns=["fold", "family","length"])
    len_sequences = []
    Name_fold = []
    Name_fam = []
    for index, row in Data.iterrows():
        Name = row['Name']
        if not Domain_bound_file.loc[Domain_bound_file['Full_Sequence_ID'] == Name].empty:
            bound =  Domain_bound_file.loc[Domain_bound_file['Full_Sequence_ID'] == Name]
            bound_start = bound['Domain_start']
            bound_end = bound['Domain_end']
            q3seq = row['q3seq'][bound_start.values[0]:bound_end.values[0]]
            rawseq = row['rawseq'][bound_start.values[0]:bound_end.values[0]]
#             print(len(row['q3seq']), Name, bound_start.values[0], bound_end.values[0], len(q3seq))
            returnDate = returnDate.append(pd.DataFrame([[Name, getattr(row, "fold"), getattr(row, "family"), q3seq, rawseq]], columns=Data.columns), ignore_index=True)
            Name_fold.append(getattr(row, "fold"))
            Name_fam.append(getattr(row, "family"))
            len_sequences.append(len(q3seq))
            df_seq = df_seq.append(pd.DataFrame([[getattr(row, "fold"), getattr(row, "family"), len(q3seq)]], columns=["fold", "family","length"]), ignore_index=True)
    return returnDate, df_seq

# Cut sequences based on teh domain bounds
def Domain_bound_cutting(Data, threshold):
    returnDate = pd.DataFrame(columns=Data.columns)
    df_seq = pd.DataFrame(columns=["fold", "family","length"])
    len_sequences = []
    Name_fold = []
    Name_fam = []
    # iterate through tables
    for index, row in Data.iterrows():
        Name = row['Name']
        # get seq length
        Seq_length = len(row['q3seq'])
        bound_start = row['Domain_start']
        bound_end = row['Domain_end']
        bound_length = bound_end - bound_start
        # if domain length less than threshold
        if bound_length <= threshold:
            # if sequence length less than threshold, direct append
            if Seq_length <= threshold:
                returnDate = returnDate.append(pd.DataFrame([[getattr(row, "Name"), getattr(row, "fold"), getattr(row, "family"), getattr(row, "Domain_start"), getattr(row, "Domain_end"), getattr(row, "q3seq"), getattr(row, "rawseq")]], columns=Data.columns), ignore_index=True)
                Name_fold.append(getattr(row, "fold"))
                Name_fam.append(getattr(row, "family"))
                len_sequences.append(Seq_length)
                df_seq = df_seq.append(pd.DataFrame([[getattr(row, "fold"), getattr(row, "family"), Seq_length]], columns=["fold", "family","length"]), ignore_index=True)
            # if sequence length longer than threshold, 
            else:
                # domain end position > threshold
                if bound_end >= threshold:
                    random_value = random.randint(bound_start - (threshold - (bound_end - bound_start)), bound_start)
                    q3seq = row['q3seq'][random_value:random_value+threshold]
                    rawseq = row['rawseq'][random_value:random_value+threshold]
                    returnDate = returnDate.append(pd.DataFrame([[getattr(row, "Name"), getattr(row, "fold"), getattr(row, "family"), getattr(row, "Domain_start"), getattr(row, "Domain_end"), q3seq, rawseq]], columns=Data.columns), ignore_index=True)
                    Name_fold.append(getattr(row, "fold"))
                    Name_fam.append(getattr(row, "family"))
                    len_sequences.append(len(q3seq))
                    df_seq = df_seq.append(pd.DataFrame([[getattr(row, "fold"), getattr(row, "family"), len(q3seq)]], columns=["fold", "family","length"]), ignore_index=True)
                #domain end position < threshold
                else:
                    random_value = random.randint(0,bound_start)
                    q3seq = row['q3seq'][random_value:random_value+threshold]
                    rawseq = row['rawseq'][random_value:random_value+threshold]                        
                    returnDate = returnDate.append(pd.DataFrame([[getattr(row, "Name"), getattr(row, "fold"), getattr(row, "family"), getattr(row, "Domain_start"), getattr(row, "Domain_end"), q3seq, rawseq]], columns=Data.columns), ignore_index=True)
                    Name_fold.append(getattr(row, "fold"))
                    Name_fam.append(getattr(row, "family"))
                    len_sequences.append(len(q3seq))
                    df_seq = df_seq.append(pd.DataFrame([[getattr(row, "fold"), getattr(row, "family"), len(q3seq)]], columns=["fold", "family","length"]), ignore_index=True)
    return  returnDate, df_seq 

#### If no domain bound information, use this to cut the sequences

def Cutting(Data, threshold):
    returnDate = pd.DataFrame(columns=Data.columns)
    df_length = pd.DataFrame(columns=["fold", "family","length"])
    len_sequences = []
    Name_fold = []
    Name_fam = []
    # iterate through tables
    for index, row in Data.iterrows():
        Name = row['Name']
        Seq_length = len(row['q3seq'])
        if Seq_length <= threshold:
            returnDate = returnDate.append(pd.DataFrame([[getattr(row, "Name"), getattr(row, "fold"), getattr(row, "family"), getattr(row, "q3seq"), getattr(row, "rawseq")]], columns=Data.columns), ignore_index=True)
            Name_fold.append(getattr(row, "fold"))
            Name_fam.append(getattr(row, "family"))
            len_sequences.append(Seq_length)
            df_length = df_length.append(pd.DataFrame([[getattr(row, "fold"), getattr(row, "family"), Seq_length]], columns=["fold", "family","length"]), ignore_index=True)

    return  returnDate, df_length 

#### Add zero padding to sequences to get them to length 798
# a helper function for two way padding

def Zero_Padding(data, maxlength):
    ReturnData = pd.DataFrame(columns = ["Name", "fold", "family", "q3seq", "rawseq", "paddings"])
    for index, row in data.iterrows():
        q3seq = ''
        rawseq = ''
        length = len(getattr(row, "q3seq"))
        tmp = '-'*(int((maxlength-length)/2))
        tmpSeq = '-'*(int((maxlength-length)/2))
        num = int((maxlength-len(row.q3seq))/2)
        if(((maxlength-len(getattr(row, "q3seq")))%2==0)):
            q3seq = tmp+getattr(row, "q3seq")+tmp
            rawseq = tmpSeq+getattr(row, "rawseq")+tmpSeq    
        else:
            q3seq = tmp+getattr(row, "q3seq")+tmp+'-'
            rawseq = tmpSeq+getattr(row, "rawseq")+tmpSeq + '-' 
        ReturnData = ReturnData.append(pd.DataFrame([[getattr(row, "Name"), getattr(row, "fold"), getattr(row, "family"), q3seq, rawseq, num]], columns = ["Name", "fold", "family", "q3seq", "rawseq", "paddings"]), ignore_index=True)
    return ReturnData

#### Partition data

def Partition(data, maxwordCount=587):
    ReturnData = pd.DataFrame(columns=['Name', 'fold', 'family', 'q3seq', 'rawseq', 'q3seqTokens', 'rawseqTokens', "paddings"])
    # iterate through the csv
    for index, row in data.iterrows():
        Name = row["Name"]
        #print(name1)
        fold = row.fold
        if len(row.q3seq) <= maxwordCount:
            q3seqTokens = list(row.q3seq)
            rawseqTokens = list(row.rawseq)
        else:
            print("Jump extra-long tokens")
        # append
        ReturnData = ReturnData.append(pd.DataFrame([[Name, fold, row.family, row.q3seq, row.rawseq, q3seqTokens,rawseqTokens, row.paddings]], columns=['Name', 'fold', 'family', 'q3seq', 'rawseq', 'q3seqTokens', 'rawseqTokens', "paddings"]), ignore_index=True)
    return ReturnData


##################################################################################

# function for RE_generator
# a helper function for mapping strings to onehot code
def Map_Tokens(data, vocab):
    indexed_tokens = []
    
    for tokens in data:
        indexed_token = []
        for token in tokens:
            if token in vocab:
                indexed_token.append(vocab[token])
        indexed_tokens.append(indexed_token)
    return indexed_tokens


# multitask dataset overwrite of Dataset
class MultitaskDatasetThree(Dataset):
    "`Dataset` for joint single and multi-label image classification."
    def __init__(self, data, labels_fold, labels_fam, paddings, cuda = True):   
        self.data = torch.FloatTensor(data.float())
        self.y_fam = torch.FloatTensor(labels_fam.float())
        self.y_fold = torch.FloatTensor(labels_fold.float())
        self.paddings = torch.FloatTensor(798-2*paddings.float())
        
        self.cuda = cuda
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self,i:int): 
        if self.cuda:
            return torch.FloatTensor(self.data[i]).float().cuda(), torch.FloatTensor([self.y_fold[i], self.y_fam[i]]).float().cuda(),  self.paddings[i].cuda()
        else:
            return torch.FloatTensor(self.data[i]).float(), torch.FloatTensor([self.y_fold[i], self.y_fam[i]]).float(), self.paddings[i]

# a helper function to load the data into custom dataset
def Dataset_Loader_Three(df, le_fam, le_fold, vocab, BATCH_SIZE, cuda = True):
    x_train = torch.LongTensor(Map_Tokens(df.q3seqTokens, vocab))
    y_train_fold = torch.LongTensor(le_fold.fit_transform(df["fold"].values.ravel()))
    y_train_fam = torch.LongTensor(le_fam.fit_transform(df["family"].values.ravel()))
    paddings = torch.LongTensor(df["paddings"].values.ravel())
    
    ds = MultitaskDatasetThree(x_train, y_train_fold, y_train_fam, paddings, cuda)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=False)
    return ds, dl


# a function to calculate reconstruction error
def reconstruction_error_calculation(model, df, le_fam, le_fold, cuda_gpu, criterion, vocab):
    gt_ds, gt_dl = Dataset_Loader_Three(df, le_fam, le_fold, vocab, BATCH_SIZE=1, cuda = cuda_gpu)
    reconstruction_err = []
    for i, data in enumerate(gt_dl, 0):
        model.eval()
        xb, yb, p = data
        output = model(xb)
        xb = xb.float()
        loss = criterion(output, xb)/(p.sum())
        reconstruction_err.append([df.iloc[i].Name,df.iloc[i].fold,df.iloc[i].family, loss.item()])
    return pd.DataFrame(reconstruction_err, columns=["Name","Fold","Family","Err"])

# a funtion to mapping model to cpu if gpu is not avaliable
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
###########################################################
## Plotting distributions

# a funtion to fit giving data using extreme value distribution
def Plot_Dist_Train_Extreme(r_err, GT_val,bin1=500,bin2=500,interval1 = 0.95,interval2=0.99):
    covMat = np.array(r_err["Err"], dtype=float)
    median = np.median(covMat)
    c, loc, scale = genextreme.fit(covMat, floc=median)
    min_extreme1,max_extreme1 = genextreme.interval(interval1,c,loc,scale)
    min_extreme2,max_extreme2 = genextreme.interval(interval2,c,loc,scale)
    x = np.linspace(min(covMat),max(covMat),2000)
    fig,ax = plt.subplots(figsize = (30,10))
    plt.xlim(0,0.4)
    plt.plot(x, genextreme.pdf(x, *genextreme.fit(covMat)), linewidth=5)
    plt.hist(np.array(r_err["Err"], dtype=float),bins=bin1,alpha=0.3,density=True,edgecolor='black',facecolor='gray', linewidth=3,histtype='stepfilled') #{'bar', 'barstacked', 'step', 'stepfilled'})
    plt.hist(np.asarray(GT_val["Err"]), bins=bin2, alpha=0.3,density=True,edgecolor='red',facecolor='red', linewidth=3,histtype='stepfilled')
    plt.xlabel('Lengths Counts')
    plt.ylabel('Probability')
    plt.title(r'max_extreme1=%.3f,max_extreme2=%.3f' %(max_extreme1, max_extreme2))
    ax.tick_params(left = False, bottom = False)
    
    ax.axvline(min_extreme1, alpha = 0.9, ymax = 0.20, linestyle = ":",linewidth=3,color="red") #,
    ax.axvline(max_extreme1, alpha = 0.9, ymax = 0.20, linestyle = ":",linewidth=3,color="red") #,
    ax.text(min_extreme1, 8, "5th", size = 20, alpha = 0.8,color="red")
    ax.text(max_extreme1, 8, "95th", size = 20, alpha =.8,color="red")
    ax.axvline(min_extreme2, alpha = 0.9, ymax = 0.20, linestyle = ":",linewidth=3,color="red") #,
    ax.axvline(max_extreme2, alpha = 0.9, ymax = 0.20, linestyle = ":",linewidth=3,color="red") #,
    ax.text(min_extreme2, 8, "1st", size = 20, alpha = 0.8,color="red")
    ax.text(max_extreme2, 8, "99th", size = 20, alpha =.8,color="red")
    
    print("95% CI upper bound:",max_extreme1)
    print("99% CI upper bound:",max_extreme2)
    print("Median RE:",np.median(np.array(GT_val["Err"], dtype=float)))
    
    return c, loc, scale, fig,ax

# Function to plot distribution of subcluster REs
def Plot_Dist_SubClust_Extreme(r_err, GT_val,bin1=100,bin2=50):
    covMat = np.array(r_err["Err"], dtype=float)
    median = np.median(covMat)
    x = np.linspace(min(covMat),max(covMat),2000)
    fig,ax = plt.subplots(figsize = (30,10))
    plt.xlim(0,1)
    plt.hist(np.array(r_err["Err"], dtype=float),bins=bin1,alpha=0.3,density=True,edgecolor='black',facecolor='gray', linewidth=3,histtype='stepfilled') #{'bar', 'barstacked', 'step', 'stepfilled'})
    plt.hist(np.asarray(GT_val["Err"]), bins=bin2, alpha=0.3,density=True,edgecolor='darkred',facecolor='red', linewidth=3,histtype='stepfilled')
    plt.xlabel('Lengths Counts')
    plt.ylabel('Probability')
    ax.tick_params(left = False, bottom = False) 
    return fig,ax