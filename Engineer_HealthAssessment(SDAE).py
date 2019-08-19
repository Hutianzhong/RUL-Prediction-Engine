import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib
from matplotlib import pyplot as plt

features = ['unit number',
            'time',
            # 'parameter 1',
            # 'parameter 2',
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


def get_dataset(path):

    data = pd.read_csv(path)
    data = np.array(data[features])
    analysis_data = data[:, 2:]
    minvalue = analysis_data.min(0).reshape(1, -1)
    maxvalue = analysis_data.max(0).reshape(1, -1)
    minvalue = minvalue.repeat(analysis_data.shape[0], axis=0)
    maxvalue = maxvalue.repeat(analysis_data.shape[0], axis=0)
    data[:, 2:] = (analysis_data - minvalue) / (maxvalue - minvalue)

    return data


def processing_data(data):

    all_data = []

    Groups_num = int(data[:, 0].max())
    for i in range(1, Groups_num+1):
        Single_data = data[data[:, 0] == i, :]
        Single_data = Single_data[:, 2:]
        Single_data = Single_data.astype('float32')
        all_data.append(Single_data)

    return all_data


# input
files_paths = './traindata/train_FD001.csv'
train_dataset = get_dataset(files_paths)
train_feature = processing_data(train_dataset)
Groups_num = len(train_feature)
n_feature = len(features[2:])
win_length = 5

# target
train_target = []
for group in range(Groups_num):
    temp = train_feature[group]
    time = np.arange(1, temp.shape[0]+1)
    time = time / time.shape[0]
    train_target.append(time)

# get batch_data
batch_x = []
batch_y = []
# for group in train_feature:
#     for t in range(group.shape[0]):
#         batch_x.append(group[t])

for group in train_feature:
    for t in range(group.shape[0] - win_length):
        batch_x.append(group[t:t+win_length])

for group in train_target:
    for t in range(win_length, group.shape[0]):
        batch_y.append(group[t])

batch_x = np.array(batch_x)
batch_y = np.array(batch_y).reshape(-1, 1)


# build sDAE_LSTM model
sDAE_LSTM = Sequential()
sDAE_LSTM.add(LSTM(32, input_shape=(win_length, n_feature), dropout=0.2))
sDAE_LSTM.add(Dense(8, activation='relu'))
sDAE_LSTM.add(Dense(4, activation='relu'))
sDAE_LSTM.add(Dense(1, activation='sigmoid'))
sDAE_LSTM.compile(optimizer='sgd', loss='mean_squared_error')

# build sDAE model
sDAE = Sequential()
sDAE.add(Dense(32, input_dim=n_feature, activation='relu'))
sDAE.add(Dense(8, activation='relu'))
sDAE.add(Dense(4, activation='relu'))
sDAE.add(Dense(1, activation='sigmoid'))
sDAE.compile(optimizer='sgd', loss='mean_squared_error')

# train sDAE model
# history = sDAE.fit(batch_x, batch_y, batch_size=100, epochs=100)
# sDAE.save_weights('sDAE_weight.h5')

# train sDAE_LSTM model
# history = sDAE_LSTM.fit(batch_x, batch_y, batch_size=100, epochs=100)
# sDAE_LSTM.save_weights('sDAE_LSTM_weight.h5')

# read weight of saved model
sDAE_LSTM.load_weights('sDAE_LSTM_weight.h5')
sDAE.load_weights('sDAE_weight.h5')

# predict
exp_no = 1
test_x = train_feature[exp_no]

pre_data = np.zeros(shape=(test_x.shape[0], 1))
pre_data[:win_length] = sDAE.predict(test_x[:win_length])

temp_x = []
for t in range(test_x.shape[0] - win_length):
    temp_x.append(test_x[t:t+win_length])
temp_x = np.array(temp_x)
pre_data[win_length:] = sDAE_LSTM.predict(temp_x)

# show in fig
plt.title('Index')
plt.plot(pre_data, "-b")
plt.show()

# smoothing data
index = np.zeros(pre_data.shape)
coe1 = 0.9
coe2 = 1 - coe1
for i in range(pre_data.shape[0]):
    if i:
        index[i] = coe1 * index[i-1] + coe2 * pre_data[i]
    else:
        index[0] = pre_data[0]

# show in fig
plt.title('Smoothing index')
plt.plot(index, "-b")
plt.savefig('./Experiment_{}/Index_SDAE_{}.jpg'.format(exp_no, exp_no))
plt.show()

pd.DataFrame(index).to_csv('index.csv', header=0, index=0)




