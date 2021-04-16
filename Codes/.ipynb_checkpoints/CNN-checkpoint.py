import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.utils.data import Dataset, DataLoader

import numpy as np

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


# channel-wise attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1   = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# spatial-wise attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv3x1(in_planes, out_planes, stride=1):
    "3x1 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BaseBlock1(nn.Module):
    
    def __init__(self, inplanes, planes, strides=1):
        super(BaseBlock1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inplanes, 
                               out_channels=planes, 
                               kernel_size=(3,3), 
                               stride=(strides,3))
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.planes = planes
    def forward(self, x):
        batch = x.shape[0]
        out = x.view(batch, 1, -1, 3)
        out = self.conv1(out)
        out = out.view(batch, self.planes, -1)
        out = self.bn(out)
        out = self.relu(out)

        
        out = self.ca(out) * out
        out = self.sa(out) * out
        

        return out

class BaseBlock2(nn.Module):
    
    def __init__(self, inplanes, planes, strides=7):
        super(BaseBlock2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inplanes, 
                               out_channels=planes, 
                               kernel_size=(7,1), 
                               stride=(strides,1))
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.inplanes = inplanes
        self.planes = planes
    def forward(self, x):
        batch = x.shape[0]
        out = x.view(batch, self.inplanes, -1, 1)
        out = self.conv1(out)
        out = out.view(batch, self.planes, -1)
        out = self.bn(out)
        out = self.relu(out)
        
        out = self.ca(out) * out
        out = self.sa(out) * out

        return out

class BaseBlock3(nn.Module):
    
    def __init__(self, inplanes, planes, strides=15):
        super(BaseBlock3, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inplanes, 
                               out_channels=planes, 
                               kernel_size=(15,1), 
                               stride=(strides,1))
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.inplanes = inplanes
        self.planes = planes
    def forward(self, x):
        batch = x.shape[0]
        out = x.view(batch, self.inplanes, -1, 1)
        out = self.conv1(out)
        out = out.view(batch, self.planes, -1)
        out = self.bn(out)
        out = self.relu(out)
        
        
        out = self.ca(out) * out
        out = self.sa(out) * out
        

        return out  
    
class CNN_Attention(nn.Module):
    def __init__(self, Fold, Fam, Prob):
        super(CNN_Attention, self).__init__()
        
        self.Keep_Probablility = Prob
        self.Fold_classes = Fold
        self.Family_classes = Fam
        
        self.layer1 = BaseBlock1(inplanes=1, planes=256)
        self.layer2 = BaseBlock2(inplanes=256, planes=512)
        self.layer3 = BaseBlock3(inplanes=512, planes=512)
#         self.layer4 = BaseBlock4(inplanes=512, planes=512)
        
        self.avgpool = nn.AdaptiveAvgPool1d(10)
        
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512,
                            kernel_size=(30, 1),
                            stride=(1, 1)),
            nn.ReLU()
        )

        self.maxpool_4 = torch.nn.Sequential(
            nn.MaxPool1d(kernel_size=(512 - 30 + 1))
        ) 
        
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512,
                            kernel_size=(50, 1),
                            stride=(1, 1)),
            nn.ReLU()
        )

        self.maxpool_5 = torch.nn.Sequential(
            nn.MaxPool1d(kernel_size=(512 - 50 + 1))
        ) 

        self.conv_6 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256,
                            kernel_size=(70, 1),
                            stride=(1, 1)),
            nn.ReLU()
        )

        self.maxpool_6 = torch.nn.Sequential(
            nn.MaxPool1d(kernel_size=(512 - 70 + 1))
        ) 
        
        self.out_fold = nn.Sequential(
            nn.Dropout(self.Keep_Probablility),
            nn.Linear(256*10, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.Keep_Probablility),
            nn.Linear(128, self.Fold_classes)
        )
        
        self.out_fam = nn.Sequential(
            nn.Dropout(self.Keep_Probablility),
            nn.Linear(256*10, 512),
            nn.ReLU(inplace=True),  
            nn.Dropout(self.Keep_Probablility),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True), 
            nn.Dropout(self.Keep_Probablility),
            nn.Linear(256, self.Family_classes)
        )   
    
    def forward(self, x):
        batch = x.shape[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature1 = x

        x = self.avgpool(x)
        
        x = x.view(batch, 1, 512, -1)
        x  = self.conv_4(x)
        x = x.view(batch, 512, -1)
        x = self.maxpool_4(x)
        
        x = x.view(batch, 1, 512, -1)
        x  = self.conv_5(x)
        x = x.view(batch, 512, -1)
        x = self.maxpool_5(x)
        
        x = x.view(batch, 1, 512, -1)
        x  = self.conv_6(x)
        x = x.view(batch, 256, -1)
        x = self.maxpool_6(x)
        feature2 = x

        out_fold = self.out_fold(x.view(batch, -1))
        out_fam = self.out_fam(x.view(batch, -1))

        return out_fold, out_fam, feature1, feature2
    
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


#define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv1d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()
        
        
def loss_batch(model, loss_func, xb, yb, opt=None, epsilon = 0.5):
    out_fold, out_fam, _, _ = model(xb)
    loss_fold = loss_func(out_fold, yb[:,0].long())
    loss_fam = loss_func(out_fam, yb[:,1].long())
    loss = epsilon*loss_fold + (1-epsilon)*loss_fam
    if opt is not None:
        if epsilon == 0:
            loss_fam.backward()
            opt.step()
            opt.zero_grad()
        elif epsilon == 1:
            loss_fold.backward()
            opt.step()
            opt.zero_grad()  
        else:
            loss.backward()
            opt.step()
            opt.zero_grad()

    return loss.item(), len(xb), loss_fold.item(), loss_fam.item()
    

def Fit(epochs, model, loss_func, opt, train_dl, valid_dl, patience=1, e = 0.5):
    writer = SummaryWriter('./log')
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    losseslist_val = []
    loss1list_val = []
    loss2list_val = []
    
    losseslist_train = []
    loss1list_train = []
    counter = 0
    loss2list_train = []

    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_dl, 0):
            xb, yb = data
            losses_train, nums_train, loss1_train, loss2_train = loss_batch(model, loss_func, xb, yb, opt, epsilon = e)
            counter = counter + 1
            if counter % 10 == 0:
                nitr = counter/10
                writer.add_scalar('Train/Loss1 ', loss1_train, nitr)
                writer.add_scalar('Train/Loss2 ', loss2_train, nitr)
                writer.add_scalar('Train/Lossall ', losses_train, nitr)
        model.eval()
        correct1 = 0
        correct2 = 0
        total = 0
        with torch.no_grad():

            losses_test, nums_test, loss1_test, loss2_test = zip(
                *[loss_batch(model, loss_func, xb, yb, opt=None, epsilon = e) for xb, yb in valid_dl]
            )

            losses_train, nums_train, loss1_train, loss2_train = zip(
                *[loss_batch(model, loss_func, xb, yb, opt=None, epsilon = e) for xb, yb in train_dl]
            )
            for xb, yb in valid_dl:
                output1, output2, _, _= model(xb)
                _, predicted1 = torch.max(output1, 1)
                _, predicted2 = torch.max(output2, 1)
                total += yb.size(0)
                correct1 += (predicted1 == yb[:,0]).sum().item()
                correct2 += (predicted2 == yb[:,1]).sum().item()

        val_loss_test = np.sum(np.multiply(losses_test, nums_test)) / np.sum(nums_test)
        val_loss1_test = np.sum(np.multiply(loss1_test, nums_test)) / np.sum(nums_test)
        val_loss2_test = np.sum(np.multiply(loss2_test, nums_test)) / np.sum(nums_test)  
        
        val_loss_train = np.sum(np.multiply(losses_train, nums_train)) / np.sum(nums_train)
        val_loss1_train = np.sum(np.multiply(loss1_train, nums_train)) / np.sum(nums_train)
        val_loss2_train = np.sum(np.multiply(loss2_train, nums_train)) / np.sum(nums_train)

        losseslist_val.append(val_loss_test)
        loss1list_val.append(val_loss1_test)
        loss2list_val.append(val_loss2_test)
        
        losseslist_train.append(val_loss_train)
        loss1list_train.append(val_loss1_train)
        loss2list_train.append(val_loss2_train)

        print("Epoch: ", epoch + 1, "Train", val_loss_train, "Validation", val_loss_test)
        print("Loss1 Train: ", val_loss1_train, "Loss1 Validation:", val_loss1_test)
        print("Loss2 Train: ", val_loss2_train, "Loss2 Validation:", val_loss2_test)

        print('Accuracy of the network on the Validation for Fold: %d %%' % (100 * correct1 / total))
        print('Accuracy of the network on the Validation for Family: %d %%' % (100 * correct2 / total))
        print("--------------------------------------------------------------------------------------\n")
        
        writer.add_scalar('Test/Accu1', correct1 / total, epoch) 
        writer.add_scalar('Test/Accu2', correct2 / total, epoch) 
        
        early_stopping(val_loss_test, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    fig, ax = plt.subplots()
    ax.plot(losseslist_val, 'b', label = "Validation")
    ax.plot(losseslist_train,'r', label = "Train")
    #plt.ylim(0,1)
    x_ticks = np.arange(0,epoch+1,1)
    plt.xticks(x_ticks)
    leg = ax.legend()
    plt.savefig("loss_fig{}_{}.png".format(epochs, e))

    return model