import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from math import sqrt

# 数据读取
data_csv = pd.read_csv('./index.csv')

# 数据预处理
data_csv = data_csv.dropna()
health_value = data_csv.values
health_value = health_value.astype('float32')
health_value = list(health_value)


# 定义模型
class lstm_one_output(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(lstm_one_output, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.layer1 = nn.Linear(hidden_size, output_size) # LSTM上层线性层
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size) # Weight学习层

    def forward(self, x):
        x, _ = self.lstm(x)  # (seq, batch, hidden)
        F.relu(x, inplace=True)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        y = self.layer1(x)
        w = F.softmax(self.layer2(x), dim=0)
        return torch.t(w).mm(y)


# 获得输入、输出数据
def get_data(dataset, win, feature, start, step):
    dataX, dataY = [], []
    for i in range(start, start+win, feature):
        a = dataset[i:(i+feature)]
        dataX.append(a)
    dataY.append(dataset[start+win+step-1])
    return np.array(dataX), np.array(dataY)


# 参数设定
data_windows = 30   # 时间窗大小
input_feature = 5   #
seq_size = data_windows//input_feature
output_size = 1
head_step = 5   # head_step预测
current_start = 150
tra_start = current_start - 50
pre_start = current_start + head_step  # 预测起始点
num_hidden = 8 # lstm网络的隐藏层节点数
num_layer = 1 # lstm网络层数
pre_list = []
real_list = []

net = lstm_one_output(input_feature, num_hidden, output_size, num_layer)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)


def training(input, real):
    net.train()
    for e in range(20):
        var_x = Variable(input)
        var_y = Variable(real)
        # 前向传播
        out = net(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def predicting(input):
    net.eval()  # 转换成测试模式
    var_x = Variable(input)
    return net(var_x)  # 返回预测结果


# 训练阶段
print("训练开始！")
for tra_point in range(tra_start, current_start, head_step):
    data_start = tra_point - data_windows - 2 * head_step + 1  # 训练数据起始点
    train_X, train_Y = get_data(health_value, data_windows, input_feature, data_start, head_step)

    train_X = train_X.reshape(-1, 1, input_feature)  # 改变shape符合模型输入
    train_Y = train_Y.reshape(-1, 1, 1)

    train_x = torch.from_numpy(train_X)  # 转成tensor
    train_y = torch.from_numpy(train_Y)

    training(train_x, train_y)
    print("完成第{}个训练点.".format(tra_point))


# 预测阶段
print("预测开始！")
for pre_point in range(pre_start, len(health_value), head_step):

    real_list.append(health_value[pre_point])
    data_start = pre_point - data_windows - 2 * head_step + 1   # 训练数据起始点
    train_X, train_Y = get_data(health_value, data_windows, input_feature, data_start, head_step)
    test_X, test_Y = get_data(health_value, data_windows, input_feature, data_start + head_step, head_step)

    train_X = train_X.reshape(-1, 1, input_feature)  # 改变shape符合模型输入
    train_Y = train_Y.reshape(-1, 1, 1)
    test_X = test_X.reshape(-1, 1, input_feature)
    test_Y = test_Y.reshape(-1, 1, 1)

    train_x = torch.from_numpy(train_X)  # 转成tensor
    train_y = torch.from_numpy(train_Y)
    test_x = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_Y)

    training(train_x, train_y)

    pre_value = predicting(test_x)
    pre_value = pre_value.view(-1).data.numpy() # 改变预测值格式
    pre_list.append(pre_value[0])
    print("完成第{}个预测点.".format(pre_point))


def error_anlysis(real, predict):
    error = []
    squaredError = []
    absError = []
    for i in range(len(real)):
        error.append(real[i]-predict[i])
        squaredError.append(error[i]**2)
        absError.append(abs(error[i]))

    result = {'Model':'Attention_LSTM'}
    result['MSE'] = sum(squaredError) / len(squaredError)
    result['RMSE'] = sqrt(sum(squaredError) / len(squaredError))
    result['MAE'] = sum(absError) / len(absError)
    print("MSE = ", result['MSE'])  # 均方误差MSE
    print("RMSE = ", result['RMSE'])  # 均方根误差RMSE
    print("MAE = ", result['MAE'])  # 平均绝对误差MAE


# 画出实际结果和预测的结果
error_anlysis(real_list, pre_list)
plt.plot(range(pre_start, len(health_value), head_step), pre_list, 'r*', label='prediction')
plt.plot(range(len(health_value)), health_value, 'b', label='real')
plt.legend(loc='best')
plt.savefig('./step{}_predict.jpg'.format(head_step))
plt.show()

