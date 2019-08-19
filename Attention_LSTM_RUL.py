import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from math import sqrt

# 数据读取
data_csv = pd.read_csv('./index.csv')

# 数据预处理
data_csv = data_csv.dropna()
health_value = data_csv.values
health_value = health_value.astype('float32')
time = np.expand_dims(np.arange(health_value.size), axis=1)
time = time.astype('float32')
max_time, min_time = time.max(), time.min()
Nor_time = list((time - min_time) / (max_time - min_time))
health_value = list(health_value)

# 健康趋势可视化
plt.title('index')
plt.plot(time, health_value, '-b')
plt.show()


# 定义模型
class lstm_one_output(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(lstm_one_output, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.layer1 = nn.Linear(hidden_size, output_size) # LSTM上层线性层
        # self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size) # Weight学习层

    def forward(self, x):
        x, _ = self.lstm(x)  # (seq, batch, hidden)
        # F.relu(x, inplace=True)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        y = self.layer1(x)
        w = F.softmax(self.layer2(x), dim=0)
        return torch.t(w).mm(y)


# 获得输入、输出数据
def get_data(time, value, win, feature, start, step):
    dataX, dataY = [], []
    for i in range(start, start+win, feature):
        a = time[i:(i+feature)]
        dataX.append(a)
    dataY.append(value[start+win+step-1])
    return np.array(dataX), np.array(dataY)


# 参数设定
data_windows = 12   # 时间窗大小
input_feature = 4   #
seq_size = data_windows//input_feature
head_step = 1   # head_step预测
current_start = 220
pre_start = current_start + head_step  # 预测起始点
# pre_start = 40
num_hidden = 32 # lstm网络的隐藏层节点数
pre_value = []
real_value = []

# 创建模型
model = lstm_one_output(input_feature, num_hidden)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
for epoch in range(100):
    loss_sum = 0
    for tra_point in range(150, current_start-data_windows):

        train_X, train_Y = get_data(Nor_time, health_value, data_windows, input_feature, tra_point, head_step)

        train_X = train_X.reshape(-1, 1, input_feature)  # 改变shape符合模型输入
        train_Y = train_Y.reshape(-1, 1, 1)

        train_x = torch.from_numpy(train_X)  # 转成tensor
        train_y = torch.from_numpy(train_Y)

        var_x = Variable(train_x)
        var_y = Variable(train_y)

        # 前向传播
        out = model(var_x)
        loss = criterion(out, var_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.data.item()
    print("完成第{}轮训练:".format(epoch))
    print("第{}轮误差为:{}".format(epoch, loss_sum))

# 模型预测
model.eval()

for pre_point in range(pre_start, len(health_value), head_step):
    real_value.append(health_value[pre_point])
    test_X, test_Y = get_data(Nor_time, health_value, data_windows, input_feature, pre_point - data_windows - head_step + 1, head_step)

    test_X = test_X.reshape(-1, 1, input_feature)
    test_Y = test_Y.reshape(-1, 1, 1)

    test_x = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_Y)

    var_x = Variable(test_x)
    pre = model(var_x)
    pre = pre.view(-1).data.numpy() # 改变预测值格式

    pre_value.append(pre[0])
    print("完成第{}个预测点:".format(pre_point))


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
error_anlysis(real_value, pre_value)
plt.plot(range(pre_start, len(health_value), head_step), pre_value, 'r*', label='prediction')
plt.plot(range(len(health_value)), health_value, '-b', label='real')
plt.legend(loc='best')
plt.savefig('./RUL_predict.jpg')
plt.show()