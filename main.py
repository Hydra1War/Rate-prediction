# %% [markdown]
# ## 1. Biblioteki

# %% [code]

import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

import os 
for dirname, _, filenames in os.walk('/kaggle/input'):# Позволяет пройти по файлам 
    for i, filename in enumerate(filenames):
        if i<5:
            print(os.path.join(dirname,filename)) # Метод join позволяет вам совместить несколько путей при помощи присвоенного разделителя


# %% [code]
# symbols = ['aapl','goog','ibm']

# %% [markdown]
# ## 2. Smotrim danye

# %% [code]
# функция которая делает дату данных по акциям
def stocks_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/{}.us.txt".format(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Close': symbol})
        df = df.join(df_temp)# С помощью  join можно делать рабочий путь до файла
    return df

dates = pd.date_range('2015-01-02','2016-12-31',freq='B')
symbols = ['goog','ibm','aapl']
df = stocks_data(symbols, dates)
df.fillna(method='pad')
print(df)
# стороим гистограмму
df.interpolate().plot()
plt.show()

# %% [code]
df.head()

# %% [code]
# строим и делаем красоту
dates = pd.date_range('2010-01-02','2017-10-11',freq='B')
df1=pd.DataFrame(index=dates)
df_ibm=pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/ibm.us.txt", parse_dates=True, index_col=0)
df_ibm=df1.join(df_ibm)
df_ibm[['Close']].plot()
plt.ylabel("stock_price")
plt.title("IBM Stock")
plt.show()

# %% [code]
df_ibm=df_ibm[['Close']]
df_ibm.info()

# %% [code]
dates = pd.date_range('2010-01-02','2017-10-11',freq='B')
df1=pd.DataFrame(index=dates)
df_aapl=pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/aapl.us.txt", parse_dates=True, index_col=0)
df_aapl=df1.join(df_aapl)
df_aapl[['Close']].plot()
plt.ylabel("stock_price")
plt.title("Apple Stock")
plt.show()

# %% [code]
df_aapl=df_aapl[['Close']]
df_aapl.info()

# %% [code]
df_ibm = df_ibm.fillna(method = 'ffill')
df_aapl = df_aapl.fillna(method = 'ffill')

# перобразует в значения от -1 до 1 (типо уменьшает по размеру)
scaler_1 = MinMaxScaler(feature_range=(-1, 1))
df_ibm['Close'] = scaler_1.fit_transform(df_ibm['Close'].values.reshape(-1,1))

scaler_2 = MinMaxScaler(feature_range=(-1, 1))
df_aapl['Close'] = scaler_2.fit_transform(df_aapl['Close'].values.reshape(-1,1))

# %% [code]
df_aapl=df1.join(df_aapl)
df_aapl[['Close']].plot()
plt.ylabel("stock_price")
plt.title("Apple Stock")
plt.show()

# %% [code]
plt.plot(df_aapl['Close'])
plt.plot(df_ibm['Close'])
plt.show()

# %% [code]
def load_data_2(stock_1 , stock_2, look_back):
    data_raw_1= stock_1.as_matrix()
    data_raw_2 = stock_2.as_matrix()
    data_useful = []
    
    # забиваем дату значениями икс-ов
    for index in range(len(data_raw_1) - look_back - 21): 
        # цикл идет до числа равного длинне data_raw минус look_back
        # look_back это на какой день мы предсказываем назад
        data_1 = []
        data_1.append(data_raw_1[index])
        data_1.append(data_raw_1[index + 7])
        data_1.append(data_raw_1[index + 14])
        data_1.extend(data_raw_1[index+20: index+21+look_back])
                
        data_2 = []  
        # добавляем сначала в новый список нужные значения
        # эти значения типо от предыдущих предсказаний
        data_2.append(data_raw_2[index])
        data_2.append(data_raw_2[index + 7])
        data_2.append(data_raw_2[index + 14])
        data_2.extend(data_raw_2[index+20: index+21+look_back])
        # делаем из списка  массив numpy
        data_2 = np.array(data_2)
        # делаем его на всякий случай -1 на 1
        data_2.reshape(-1,1)
        data_1 = np.array(data_1)
        data_1.reshape(-1,1)
        data_3 = np.hstack((data_1 , data_2))
        # "приклеиваем" дата 2 к основной дате
        data_useful.append(data_3)
        if index == -1:
            print(type(data_raw_1[index:index + look_back]))
            print(data_raw_1[index:index + look_back].shape)
    # делаем из списка  массив numpy
    data_useful = np.array(data_useful);

    test_set_size = int(np.round(0.3 * data_useful.shape[0]));
    train_set_size = data_useful.shape[0] - (test_set_size);
    
    x_train_useful = data_useful[:train_set_size,:-1,:]
    y_train_useful = data_useful[:train_set_size,-1,:]
    
    x_test_useful  = data_useful [train_set_size:,:-1]
    y_test_useful  = data_useful [train_set_size:,-1,:]
    
    return [x_train_useful , y_train_useful , x_test_useful , y_test_useful]

# %% [code]
look_back = 10
x_train_useful, y_train_useful, x_test_useful, y_test_useful = load_data_2(df_aapl, df_ibm, look_back)
print('x_train_useful.shape = ',x_train_useful.shape)
print('y_train_useful.shape = ',y_train_useful.shape)
print('x_test_useful.shape = ',x_test_useful.shape)
print('y_test_useful.shape = ',y_test_useful.shape)


# %% [code]
# Делаем тензоры 
x_train_useful = torch.from_numpy(x_train_useful).type(torch.Tensor)
x_test_useful = torch.from_numpy(x_test_useful).type(torch.Tensor)
y_train_useful = torch.from_numpy(y_train_useful).type(torch.Tensor)
y_test_useful = torch.from_numpy(y_test_useful).type(torch.Tensor)

print(x_train_useful.size(), y_train_useful.size())

n_steps = look_back-1
batch_size = 1606
num_epochs = 350
#n_iters = 3000
#n_iters / (len(train_X) / batch_size)
#num_epochs = int(num_epochs)

# train_useful = torch.utils.data.TensorDataset(x_train_useful,y_train_useful)
# test_useful = torch.utils.data.TensorDataset(x_test_useful,y_test_useful)

# train_loader_useful = torch.utils.data.DataLoader(dataset=train_useful, 
#                                            batch_size=batch_size, 
#                                            shuffle=False)

# test_loader_useful = torch.utils.data.DataLoader(dataset=test_useful, 
#                                           batch_size=batch_size, 
#                                           shuffle=False)

# %% [markdown]
# ## 3. Sozdayem model

# %% [code]
#####################
input_dim = 2
hidden_dim = 32
num_layers = 2 
output_dim = 2

# создаем слои самой нейронной сети
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # магия
        self.hidden_dim = hidden_dim

        # Количество скрытых слоев
        self.num_layers = num_layers

        # Создаем нашу нейросеть тип LSTM
        # batch_first=True так надо
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # рандомно отключаем нейроны для лучшей обучаемости
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # функция для отеключения нейроноа
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :]) 
        
        return out
    
model = LSTM(input_dim = input_dim, hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers)

loss_fn = torch.nn.MSELoss(size_average=True)

optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# %% [code]
# optimiser = torch.optim.Adam(model.parameters(), lr=0.00001)

# %% [code]
# Train model
#####################
# Build model
#####################
input_dim = 2
hidden_dim = 32
num_layers = 2 
output_dim = 2
num_epochs = 1000
hist = np.zeros(num_epochs)

# главная функция и обучения(цикл) 
seq_dim =look_back-1  
model.train()
for t in range(num_epochs):
    # model.hidden = model.init_hidden()
    
    # print(type(x_train_aapl))
    y_train_pred = model(x_train_useful)

    loss = loss_fn(y_train_pred, y_train_useful)
    
    print(loss)
    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # функция для обнуление градиента(сложно объяснить, но мы градиентом уменьшаем ошибку) 
    optimiser.zero_grad()
    loss.backward()

    # тоже сложная функция оптимизации
    optimiser.step()

# %% [code]
plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

# %% [code]
# model.train(False)

# %% [code]

# Cтроим для apple train
plt.plot(y_train_pred.detach().numpy()[:,0], label="Apple")
plt.plot(y_train_useful.detach().numpy()[:,0])
plt.legend()
plt.show()

# %% [code]
plt.plot(y_train_pred.detach().numpy()[:,1], label="Ibm")
plt.plot(y_train_useful.detach().numpy()[:,1])
plt.legend()
plt.show()

# %% [code]
np.shape(y_train_pred)

# %% [code]
y_train_useful.detach().numpy().shape

# %% [code]
y_train_pred = model(x_train_useful)
y_test_pred = model(x_test_useful)

# %% [code]
print(y_train_pred.shape)
print(y_test_pred.shape)

# %% [code]
# Строим для apple test
plt.plot(y_test_pred.detach().numpy()[:,0], label="Apple_pred")
plt.plot(y_test_useful.detach().numpy()[:,0] , label="True")
plt.legend()
plt.show()

# %% [code]
# Строим для IBM test
plt.plot(scaler_2.inverse_transform(y_test_pred.detach().numpy()[:,1].reshape(-1, 1)), label="Ibm_pred")
plt.plot(scaler_2.inverse_transform(y_test_useful.detach().numpy()[:,1].reshape(-1, 1)), label="True")
plt.legend()
plt.show()

# %% [code]
# разделяем на aapl и ibm
y_train_pred_aapl = y_train_pred[:,1]
y_train_pred_ibm = y_train_pred[:,0]
y_test_pred_aapl = y_test_pred[:,1]
y_test_pred_ibm = y_test_pred[:,0]

# делаем reshape
y_train_useful[:,1].detach().numpy().reshape(-1, 1).shape
y_train_pred_aapl.reshape(-1, 1).shape

# %% [code]
y_test_pred_aapl.size()

# %% [code]
y_train_pred_aapl.shape

# %% [code]
y_test_pred_aapl.shape

# %% [code]
y_train_pred[:2]

# %% [code]
y_train_useful[:2]

# %% [code]
y_train_pred_aapl[:2]

# %% [code]
########################### apple
y_train_pred_aapl = scaler_2.inverse_transform(y_train_pred_aapl.detach().numpy().reshape(-1, 1))
y_test_pred_aapl = scaler_2.inverse_transform(y_test_pred_aapl.detach().numpy().reshape(-1, 1))

# Считаем RMSE корень из средней квадратичной ошибки
trainScore_aapl = math.sqrt(mean_squared_error(scaler_2.inverse_transform(y_train_useful[:,1].detach().numpy().reshape(-1, 1)) , y_train_pred_aapl.reshape(-1, 1)))
print('Train Score Apple: %.2f RMSE' % (trainScore_aapl))
testScore_aapl = math.sqrt(mean_squared_error(scaler_2.inverse_transform(y_test_useful[:,1].detach().numpy().reshape(-1, 1)), y_test_pred_aapl.reshape(-1, 1) ))
print('Test Score Apple: %.2f RMSE' % (testScore_aapl))

########################### ibm
y_train_pred_ibm = scaler_1.inverse_transform(y_train_pred_ibm.detach().numpy().reshape(-1, 1))
y_test_pred_ibm = scaler_1.inverse_transform(y_test_pred_ibm.detach().numpy().reshape(-1, 1))

# Считаем RMSE корень из средней квадратичной ошибки
trainScore_ibm = math.sqrt(mean_squared_error(scaler_1.inverse_transform(y_train_useful[:,1].detach().numpy().reshape(-1, 1)), y_train_pred_ibm.reshape(-1, 1)))
print('Train Score Ibm: %.2f RMSE' % (trainScore_ibm))
testScore_ibm = math.sqrt(mean_squared_error(scaler_1.inverse_transform(y_test_useful[:,1].detach().numpy().reshape(-1, 1)) , y_test_pred_ibm.reshape(-1, 1) ))
print('Test Score Ibm: %.2f RMSE' % (testScore_ibm))

# %% [code]
scaler_1.inverse_transform(y_test_useful[:,1].detach().numpy().reshape(-1, 1))[:5]

# %% [code]
y_test_pred_ibm.reshape(-1, 1)[:5]

# %% [code]
trainPredictPlot = np.empty_like(df_aapl)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back + 21:len(y_train_pred_aapl)  + 21 +look_back, :] = y_train_pred_aapl

print(trainPredictPlot)
testPredictPlot = np.empty_like(df_aapl)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred_aapl)+ 21 +look_back-1:len(df_aapl)-1, :] = y_test_pred_aapl

# делаем график(Большой)
plt.figure(figsize=(16,8))
plt.plot(scaler_2.inverse_transform(df_ibm))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
plt.show()


# %% [code]
trainPredictPlot = np.empty_like(df_ibm)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back + 21:len(y_train_pred_aapl)+look_back +21 , :] = y_train_pred_ibm

print(trainPredictPlot)
# готовим данные для графика
testPredictPlot = np.empty_like(df_ibm)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred_ibm)+ 21 +look_back-1:len(df_ibm)-1, :] = y_test_pred_ibm

# опять делаем график
plt.figure(figsize=(16,8))
plt.plot(scaler_1.inverse_transform(df_aapl))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
plt.show()
