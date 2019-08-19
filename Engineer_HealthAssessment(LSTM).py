import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
import torch
import matplotlib


features = ['unit number',
            'time',
            'parameter 1',
            'parameter 2',
            # 'parameter 3',
            # 'parameter 4',
            'parameter 5',
            'parameter 6',
            'parameter 7',
            # 'parameter 8',
            # 'parameter 9',
            'parameter 10',
            'parameter 11',
            'parameter 12',
            # 'parameter 13',
            'parameter 14',
            'parameter 15',
            'parameter 16',
            'parameter 17',
            'parameter 18',
            # 'parameter 19',
            'parameter 20',
            # 'parameter 21',
            # 'parameter 22',
            'parameter 23',
            'parameter 24']


def get_all_files(file_dir):

    files = []

    train_list = os.listdir(file_dir)

    for i in range(len(train_list)):
        path = os.path.join(file_dir, train_list[i])
        files.append(path)

    return files


def get_dataset(path):

    data = pd.read_csv(path)
    data = np.array(data[features])
    analysis_data = data[:, 2:]
    data[:, 2:] = (analysis_data - analysis_data.mean(axis=0)) / analysis_data.std(axis=0)
    # data = data[:, np.isnan(data[0, :]) == False]

    return data


def xlsx_to_csv(files_paths):

    for xls_path in files_paths:
        data_xls = pd.read_excel(xls_path)
        csv_path = xls_path.replace('xlsx', 'csv')
        data_xls.to_csv(csv_path, encoding='utf-8')


def processing_data(data):

    all_data = []

    Groups_num = int(data[:, 0].max())
    for i in range(1, Groups_num+1):
        Single_data = data[data[:, 0] == i, :]
        Single_data = Single_data[:, 2:]
        Single_data = Single_data.astype('float32')
        all_data.append(Single_data)

    return all_data


# files_paths = get_all_files(file_dir='./traindata')

# xlsx_to_csv(files_paths)

files_paths = './traindata/train_FD001.csv'

train_dataset = get_dataset(files_paths)

train_dataset = processing_data(train_dataset)

Groups_num = len(train_dataset)


# 假设每组数据前health_point为正常数据
health_point = 30
health_data = []
for data in train_dataset:
    health_data.append(data[:health_point, :])
win_length = 10

batch_data = []
for group in health_data:
    for point in range(health_point-win_length+1):
        batch_data.append(group[point:point+win_length])

batch_data = np.array(batch_data)


# 定义模型
class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_AE, self).__init__()

        self.en_lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)  # 编码
        # self.relu = nn.ReLU()
        self.de_lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)   # 解码
        self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.en_lstm(x)
        x, _ = self.de_lstm(x)
        x = self.layer(x)
        return x


# 参数设置
batch_num, _, n_features = batch_data.shape
hidden_num = 32
epo = 1000
train_timestep = 100
batch_size = 100

# 创建模型
net = LSTM_AE(n_features, hidden_num, n_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 训练模型
all_loss = torch.zeros(epo)
for epoch in range(epo):
    rand_select = np.arange(batch_data.shape[0])
    np.random.shuffle(rand_select)
    x = batch_data[rand_select[0:batch_size]]
    x = torch.from_numpy(x)
    var_x = Variable(x)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_x)
    all_loss[epoch].add_(loss)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 可视化训练过程
    if epoch % train_timestep == 0:
        print("完成{}/{}, loss = {}".format(epoch, epo, all_loss[epoch].item()))


# loss可视化
plt.title('loss')
plt.plot(all_loss.data.numpy(), "-b")
plt.show()

# 模型预测
exp_no = 95
real_data = train_dataset[exp_no]
rec_data = np.zeros(real_data.shape)
net = net.eval()


def predict_data(x):
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    var_x = Variable(x)
    out = net(var_x)
    out = out.view(win_length, n_features).data.numpy()
    return out


rec_data[:win_length] = predict_data(real_data[:win_length])

for test_point in range(win_length, real_data.shape[0]):
    pre_data = predict_data(real_data[test_point-win_length+1:test_point+1])
    rec_data[test_point] = pre_data[-1, :]


# 重构数据可视化
for i, fea in enumerate(features[2:]):
    plt.title(fea)
    plt.plot(real_data[:, i], "-b")
    plt.plot(rec_data[:, i], "-r")
    # plt.savefig('./Experiment_{}/parameter_{}.jpg'.format(exp_no, i))
    plt.show()

# 多元高斯分布建模
index = np.zeros(real_data.shape[0])
error_data = np.abs(real_data - rec_data)
# ave = np.mean(error_data, axis=1).reshape(-1, 1)
ave = np.mean(error_data, axis=0)
cov = np.cov(error_data, rowvar=False)
inverse_cov = np.linalg.inv(cov)

for i in range(real_data.shape[0]):
    temp = (error_data[i] - ave).reshape(1, -1)
    # index[i] = np.dot(np.dot(temp, inverse_cov), temp.T).reshape(-1)
    index[i] = np.dot(np.dot(temp, cov), temp.T).reshape(-1)


def autoNorm(x):
    minvals = x.min()
    maxvals = x.max()
    ranges = maxvals - minvals
    norm = (x - minvals) / ranges
    return norm


# 可视化性能指数
index = autoNorm(index)
# threshold = np.mean(index)
threshold = 0.02
mark = index > threshold
label = np.zeros(index.shape)
label[mark] = 1
plt.title('Index')
plt.scatter(np.arange(index.shape[0]), index, c=label, cmap=matplotlib.colors.ListedColormap(['blue', 'red']))
# plt.savefig('./Experiment_{}/index_{}.jpg'.format(exp_no, exp_no))
plt.show()

