#coding=gbk
import torch
import torch.nn as nn
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # 标准化
import os

file_path=r"D:\program\matlab\bin\dataset"  #matlab生成文件路径
Tolerable_relative_error=0.2                #可容忍心最大相对误差


def create_inout_sequences(file_path):
    inout_seq = []
    file_list = os.listdir(file_path)
    for i in range(len(file_list)):
        i=i+1
        fid=open(file_path+"/"+str(i)+".txt")
        source_in_lines = fid.readlines()
        fid.close()
        data=source_in_lines[0].split(" ")
        for j,value in enumerate(data[:-1]):
            data[j] = float(value)
        train_seq = np.array(data[2:-1])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data_normalized = scaler.fit_transform(train_seq.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

        train_label = np.array([(data[0]-0.3)*5.0,(data[1]-1.4)*1.6])
        train_label = torch.from_numpy(train_label)
        inout_seq.append((train_data_normalized ,train_label))
    return inout_seq


#separate train set and test set
inout_seq=create_inout_sequences(file_path)
total_num=len(inout_seq)
train_inout_seq=inout_seq[:int(0.7*total_num)]
test_inout_seq=inout_seq[-int(0.3*total_num):]
print("Train set "+str(len(train_inout_seq))+" files"+"      "+"Test set "+str(len(test_inout_seq))+" files")

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size                  # 隐藏层节点数100
        self.lstm = nn.LSTM(input_size, hidden_layer_size)          #输入维度1，隐藏层100

        self.linear = nn.Linear(hidden_layer_size, output_size)     #隐藏层与全连接层相连，作为输出

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]                                      #输出预测呼吸速率，心率值


model = LSTM()
loss_function = nn.MSELoss()                                        #损失函数为MSE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)          #Adam优化器，学习率0.001



#model train
epochs = 300
loss_list=[]
for i in range(epochs):
    loss_list_epoch = []
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        labels= labels.to(torch.float32)
        y_pred = y_pred.to(torch.float32)
        single_loss = loss_function(y_pred, labels)     #损失函数
        loss_list_epoch.append(single_loss.item())
        single_loss.backward()                          #前向传播
        optimizer.step()
    mean_loss_epoch=sum(loss_list_epoch)/len(loss_list_epoch)
    loss_list.append(mean_loss_epoch)
    if i%10 == 0:
        print(f'epoch: {i:3}    loss: {mean_loss_epoch:10.8f}')
epoch_list=np.arange(0,len(loss_list))
loss_list=np.array(loss_list)
plt.plot(epoch_list,loss_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#model test
model.eval()

i=0
j=0
for seq, labels in train_inout_seq:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        y_pred=y_pred.numpy()
        labels = labels.numpy()
        if  abs(y_pred[0]-labels[0])/labels[0]<=Tolerable_relative_error and abs(y_pred[1]-labels[1])/labels[1]<=Tolerable_relative_error:
            j=j+1
    i=i+1
accuracy=j/i
print("Train set accuracy: "+str(accuracy))


i=0
j=0
for seq, labels in test_inout_seq:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        y_pred=y_pred.numpy()
        labels = labels.numpy()
        if  abs(y_pred[0]-labels[0])/labels[0]<=Tolerable_relative_error and abs(y_pred[1]-labels[1])/labels[1]<=Tolerable_relative_error:
            j=j+1
    i=i+1
accuracy=j/i
print("Test set accuracy: "+str(accuracy))




