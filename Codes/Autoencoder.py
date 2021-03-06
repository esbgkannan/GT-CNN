from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt

class autoencoder(nn.Module):
    def __init__(self, model):
        super(autoencoder, self).__init__()
        # pretrained cnn model
        self.encoder = model
        for p in self.parameters():
            p.requires_grad = False
            
        # decoder
        
        self.trans1 = nn.ConvTranspose1d(in_channels=512, 
                                         out_channels=512, 
                                         kernel_size=15, 
                                         stride=15, 
                                         dilation=2, 
                                         padding=3)
        
        self.relu1 = torch.nn.ReLU(inplace=True)
        
        self.in1 = nn.InstanceNorm1d(512)
        
        self.trans2 = nn.ConvTranspose1d(in_channels=512, 
                                         out_channels=512, 
                                         kernel_size=7, 
                                         stride=7,
                                         dilation=2, 
                                         padding=1, 
                                         output_padding=1)

        self.relu2 = torch.nn.ReLU(inplace=True)   
        
        self.in2 = nn.InstanceNorm1d(512)

        self.trans3 = nn.ConvTranspose1d(in_channels=512, 
                                         out_channels=3, 
                                         kernel_size=3, 
                                         stride=1) 
        

    def forward(self, x):
        _, _, x, _ = self.encoder(x)
        batch = x.shape[0]
        x = self.trans1(x)
        x = self.relu1(x)
        x = self.in1(x)
        x = self.trans2(x)
        x = self.relu2(x) 
        x = self.in2(x)
        x = self.trans3(x)
        x = F.sigmoid(x)  
        x = x.transpose(1,2)
        return x
    
def fit_autoencoder(epochs, model, criterion, optimizer, train_dl, val_dl_test, val_dl, patience = 5):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    train_loss = 0.0
    val_loss = 0.0
    val_loss_test = 0.0
    val_loss_oof = 0.0
    train_loss_list = []
    val_loss_list = []
    val_loss_test_list = []
    val_loss_oof_list = []
    for epoch in range(epochs):
        
        model.train()
        for i, data in enumerate(train_dl, 0):
            xb, yb, p = data
            optimizer.zero_grad()
            output = model(xb)
            xb = xb.float()
            loss = criterion(output, xb)/(p.sum())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*xb.size(0)
        model.eval()
    
        with torch.no_grad():
            for i, data in enumerate(val_dl, 0):
                xb, yb, p = data
                output = model(xb)
                xb = xb.float()
                loss = criterion(output, xb)/(p.sum())
                val_loss += loss.item()*xb.size(0)
        
        with torch.no_grad():
            for i, data in enumerate(val_dl_test, 0):
                xb, yb, p = data
                output = model(xb)
                xb = xb.float()
                loss = criterion(output, xb)/(p.sum())
                val_loss_test += loss.item()*xb.size(0)
                
        train_loss = train_loss/len(train_dl)
        val_loss = val_loss/len(val_dl)
        val_loss_test = val_loss_test/len(val_dl_test)
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_loss_test_list.append(val_loss_test)
        
        print("Epoch :{} \tTraining Loss :{:.6f}.".format(epoch+1,train_loss))
        
        print("Epoch :{} \tVal Loss :{:.6f}.".format(epoch+1,val_loss))
        
        print("Epoch :{} \tVal OOD Loss :{:.6f}.".format(epoch+1,val_loss_test))

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        train_loss = 0.0
        val_loss = 0.0
    fig, ax = plt.subplots()
    ax.plot(val_loss_list, 'b', label = "Validation")
    ax.plot(train_loss_list,'r', label = "Train")
    ax.plot(val_loss_test,'y', label = "Validation Lyso")
    x_ticks = np.arange(0,epoch+1,1)
    plt.xticks(x_ticks)
    
    return model

# EarlyStopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
