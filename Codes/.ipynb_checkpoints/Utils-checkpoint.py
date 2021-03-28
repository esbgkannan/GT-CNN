import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.utils.data import *
from torch.utils.data import Dataset, DataLoader

import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import RandomOverSampler

from collections import OrderedDict
import json
import subprocess
import sys
import time
import xml.etree.ElementTree

# multitask dataset overwrite of Dataset
class MultitaskDataset(Dataset):
    "`Dataset` for joint single and multi-label image classification."
    def __init__(self, data, labels_fold, labels_fam, cuda = True):   
        self.data = torch.FloatTensor(data.float())
        self.y_fam = torch.FloatTensor(labels_fam.float())
        self.y_fold = torch.FloatTensor(labels_fold.float())
        self.cuda = cuda
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self,i:int): 
        if self.cuda:
            return torch.FloatTensor(self.data[i]).float().cuda(), torch.FloatTensor([self.y_fold[i], self.y_fam[i]]).float().cuda()
        else:
            return torch.FloatTensor(self.data[i]).float(), torch.FloatTensor([self.y_fold[i], self.y_fam[i]]).float()

# a helper function to load the data into custom dataset
def Dataset_Loader(df, le_fam, le_fold, vocab, BATCH_SIZE, cuda = True):
    x_train = torch.LongTensor(Map_Tokens(df.q3seqTokens, vocab))
    y_train_fold = torch.LongTensor(le_fold.fit_transform(df["fold"].values.ravel()))
    y_train_fam = torch.LongTensor(le_fam.fit_transform(df["family"].values.ravel()))
    
    ds = MultitaskDataset(x_train, y_train_fold, y_train_fam, cuda)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False)
    return ds, dl

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

# a helper function for read txt
def List2String(s):  
    # initialize an empty string 
    str1 = ""  
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    # return string   
    return str1  

# a helper function for data Normalization in range (0,1)
def Normalization(data, show = False):
    _range = np.max(data) - np.min(data)
    if show:
        x = np.linspace(1,len(data),len(data))
        plt.plot(x,data, color='red')
        plt.show()
    return (data - np.min(data)) / _range

# a helper function for data Standardization with mean 0 and sigma as std
def Standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# a function to caldulate reconstruction error
def reconstruction_error_calculation(model, df, le_fam, le_fold, cuda_gpu, criterion):
    gt_ds, gt_dl = Dataset_Loader(df, le_fam, le_fold, vocab, BATCH_SIZE=1, cuda = cuda_gpu)
    reconstruction_err = []
    for i, data in enumerate(gt_dl, 0):
        model.eval()
        xb, yb, p = data
        output = model(xb)
        xb = xb.float()
        loss = criterion(output, xb)/(p.sum())
        reconstruction_err.append([df.iloc[i].Name,df.iloc[i].fold,df.iloc[i].family, loss.item()])
    return pd.DataFrame(reconstruction_err, columns=["Name","Fold","Family","Err"])

# a funtion to fit giving data using extreme value distribution
def Plot_Len_Dis_Extreme(r_err, GT_val, interval1 = 0.95, interval2 = 0.99, bins_val=100):
    covMat = np.array(r_err["Err"], dtype=float)
    median = np.median(covMat)
    c, loc, scale = genextreme.fit(covMat, floc=median)

    min_extreme1,max_extreme1 = genextreme.interval(interval1,c,loc,scale)
    min_extreme2,max_extreme2 = genextreme.interval(interval2,c,loc,scale)
    
    x = np.linspace(min(covMat),max(covMat),2000)

    fig,ax = plt.subplots(1, 1)
    plt.xlim(0,0.4)
    plt.plot(x, genextreme.pdf(x, *genextreme.fit(covMat)))
    plt.hist(np.array(r_err["Err"], dtype=float),bins=100,alpha=0.7, density=True)
    plt.hist(np.asarray(GT_val["Err"]), edgecolor='k', alpha=0.35, bins=bins_val, density=True) 
    plt.xlabel('Lengths Counts')
    plt.ylabel('Probability')
    plt.title(r'max_extreme1=%.3f,max_extreme2=%.3f' %(max_extreme1, max_extreme2))
    plt.annotate('Max Extreme Value 1',xy=(max_extreme1,0),xytext=(max_extreme1,1),arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color="black"),color="black")
    plt.annotate('Max Extreme Value 2',xy=(max_extreme2,0),xytext=(max_extreme2,1),arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color="black"),color="black")
    plt.grid(True)
    median = GT_val.median()
    print("95% CI upper bound:",max_extreme1)
    print("99% CI upper bound:",max_extreme2)
    print("Median RE:",median.values)
    return max_extreme1, max_extreme2, median

# a funtion to mapping model to cpu if gpu is not avaliable
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
# a helper function to load dataset and get input length
def Load_data(file, Fam_threshold):
    df_return = pd.read_csv(file)    
    print(len(df_return))
    Seq_length = len(df_return.iloc[0].q3seq)
    selected_fam = (df_return.groupby('family').count().name)\
    [(df_return.groupby('family').count().name) > Fam_threshold]\
    .index.tolist()
    df_return = df_return[df_return['family'].isin(selected_fam)]
    print(df_return.info())
    print(df_return.groupby('fold').count())
    print(len(df_return))
    return  df_return, Seq_length

# a helper function for train test validation split
def Train_Test_Val_split(df, strantify_type="fold", split_rate=0.1, random_state=2020):
    train_df, test_df = train_test_split(df, test_size=split_rate, random_state=random_state, 
                               stratify=df[strantify_type])
    train_df, val_df = train_test_split(train_df, test_size=(split_rate)/(1-split_rate), random_state=random_state, 
                               stratify=train_df[strantify_type])
    print(str(len(train_df))+" "+str(len(test_df))+" "+str(len(val_df)))
    return train_df, test_df, val_df

# a helper function to balance between families, the families above the threshold will be cut in half. This method is deprecated in later process.
def Family_Balance(df, Fam_threshold):
    selected_fam_above = (df.groupby('family').count().Name)[(df.groupby('family').count().Name) < Fam_threshold].index.tolist()
    df_return_1 = df[df['family'].isin(selected_fam_above)]
    selected_fam_below = (df.groupby('family').count().Name)[(df.groupby('family').count().Name) > Fam_threshold].index.tolist()
    df_return_2 = df[df['family'].isin(selected_fam_below)]
    df_return_2,_ = train_test_split(df_return_2, test_size=0.5, random_state=2020, 
                                   stratify=df_return_2['family'])
    df_return = pd.concat([df_return_1, df_return_2])
    return df_return

# a helper function to trim the data based on threshold, only families that are larger than threshold will be selected.
def Trim_data(df, Fam_threshold):
    df_return = df
    selected_fam = (df_return.groupby('family').count().Name)\
    [(df_return.groupby('family').count().Name) > Fam_threshold]\
    .index.tolist()
    df_return = df_return[df_return['family'].isin(selected_fam)]
    return  df_return


"""
This function will implement salt and pepper noise on sequences, the SNR rate controls how much will the noise
be imposed on RAW SS area. 
Step 1: Make a copy of the original sequence
Step 2: Make a mask based on SNR to determine whether the pixel is the original signal or noise
Step 3: Give the original image a noise value according to the mask
"""
def Salt_Pepper_Noise(df, Fam_threshold, SNR):
    selected_fam_above = (df.groupby('family').count().Name)[(df.groupby('family').count().Name) >= Fam_threshold].index.tolist()
    df_return_1 = df[df['family'].isin(selected_fam_above)]
    selected_fam_below = (df.groupby('family').count().Name)[(df.groupby('family').count().Name) < Fam_threshold].index.tolist()
    df_selected = df[df['family'].isin(selected_fam_below)]
    family_list = df_selected.groupby('family').count().Name
    df_noised = pd.DataFrame().reindex_like(df_selected)
    while min(family_list)<Fam_threshold:
        random_position = np.random.choice(len(df_selected))
        row = df_selected.iloc[random_position]
        
        seq_name = row["Name"]
        seq_ = list(row.q3seq[row.paddings:-row.paddings])
        seq_len = len(seq_)
        if family_list[row.family] < Fam_threshold:# if family size still smaller than threshold, then add noise and augmentation

            mask = np.random.choice((0, 1), size=seq_len, p=[SNR, (1 - SNR)])
            for i in range(len(mask)):
                if mask[i]==1:
                    seq_[i] = 'C'
            new_seq = list(row.q3seq[:row.paddings]) + seq_ + list(row.q3seq[-row.paddings:])
            family_list[row.family] = family_list[row.family]+1
            new_seq = ''.join(new_seq)
            df_noised = df_noised.append(pd.DataFrame([[seq_name+'_'+str(family_list[row.family]), row.fold, row.family, new_seq, row.rawseq, list(new_seq),list(row.rawseq), row.paddings]], columns=['Name', 'fold', 'family', 'q3seq', 'rawseq', 'q3seqTokens', 'rawseqTokens', "paddings"]), ignore_index=True)
    df_return = pd.concat([df_return_1, df_selected, df_noised], ignore_index=True)
    df_return.drop_duplicates(['q3seq'], inplace=True, ignore_index=True)
    df_return.dropna(axis=0, how='any', inplace=True)
    return df_return  

# a helper function for oversampling with random oversampler, this sampler will rechoice small classes.
def OverSampling(df):
    data  = df.q3seq.to_numpy()
    ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
    X_res, y_res = ros.fit_resample(data.reshape(-1,1), df.family)

    dataset = pd.DataFrame()
    dataset['family'] = y_res.tolist()
    dataset['X_res'] = X_res.tolist()

    df_return = df.iloc[ros.sample_indices_]
    return df_return


# a helper function to plot confuction matrix
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.grid(b=True)
    plt.plot()
    
# Evaluation the model
def model_evaluation(model, le_fam, le_fold, Test_dl, cm = "fold"):
    try:
        model.eval()
        with torch.no_grad():
            _predictions_1 = []
            _predictions_2 = []
            _gt_1 = []
            _gt_2 = []
            for xb, yb in Test_dl:
                output1, output2, _ = model(xb)
                _, predicted1 = torch.max(output1, 1)
                _, predicted2 = torch.max(output2, 1)
                _predictions_1.extend(predicted1.cpu().numpy())
                _predictions_2.extend(predicted2.cpu().numpy())
                _gt_1.extend(yb[:,0].cpu().numpy())
                _gt_2.extend(yb[:,1].cpu().numpy())

            _predictions_1 = le_fold.inverse_transform(_predictions_1)
            _predictions_2 = le_fam.inverse_transform(_predictions_2)
            #print(_gt_2)
            _gt_1 = list(map(int, _gt_1))
            _gt_1 = le_fold.inverse_transform(_gt_1)

            _gt_2 = list(map(int, _gt_2))
            _gt_2 = le_fam.inverse_transform(_gt_2)

            print(classification_report(_gt_1, _predictions_1))
            print(classification_report(_gt_2, _predictions_2))
            #plot confusion matrix
            if cm == "fold":
                cm1 = plot_confusion_matrix(confusion_matrix(_gt_1, _predictions_1), df_large_zero["fold"].unique(), "Confusion Matrix Fold")
            else:
                cm2 = plot_confusion_matrix(confusion_matrix(_gt_2, _predictions_2), df_large_zero["family"].unique(), "Confusion Matrix Family")
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise exception

# a helper function to cut the curve into sequences without padding
def Curve_Cut(data, window=5):
    newdata = []
    length = len(data)
    for i in range(length-window):
        sum = 0
        for x in data[i:i+window]:
            sum +=x
        avg = sum/window
        if abs(data[i]-avg)>0.02:
            newdata.append(data[i])
    return np.asarray(newdata)

# a helper function to cut the curve into sequences without padding
def Curve_Cut_Paddings(data, paddings):
    newdata = []
    length = len(data)
    newdata = data[paddings:(length-paddings)]
    return np.asarray(newdata)


# a helper function using savgol trnsformation to smoothing the data
def Savgol_Transform(data, window_size, show=False):
    yhat = scipy.signal.savgol_filter(data, window_size, 4) # window size 51, polynomial order 3
    x = np.linspace(1,len(data),len(data))
    if show:
        plt.plot(x,yhat, color='red')
        plt.show()
    return yhat



def Print_GPU():
    def extract(elem, tag, drop_s):
      text = elem.find(tag).text
      if drop_s not in text: raise Exception(text)
      text = text.replace(drop_s, "")
      try:
        return int(text)
      except ValueError:
        return float(text)

    i = 0

    d = OrderedDict()
    d["time"] = time.time()

    cmd = ['nvidia-smi', '-q', '-x']
    cmd_out = subprocess.check_output(cmd)
    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

    util = gpu.find("utilization")
    d["gpu_util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / 11171

    if d["gpu_util"] < 15 and d["mem_used"] < 2816 :
        msg = 'GPU status: Idle \n'
    else:
        msg = 'GPU status: Busy \n'

    now = time.strftime("%c")
    print('\n\nUpdated at %s\n\nGPU utilization: %s %%\nVRAM used: %s %%\n\n%s\n\n' % (now, d["gpu_util"],d["mem_used_per"], msg))