import pandas as pd
import numpy as np
import math
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import yfinance as yfin
import datetime 

start = datetime.datetime(2012,1,1)
end = datetime.datetime(2023,10,25)

df = yfin.download(tickers=['BTC-USD'], start=start,end=end,auto_adjust = True)
df
plt.style.use('fivethirtyeight')
plt.figure(figsize = (16,8))
plt.title('Preco de fechamento - GOOGLE')
plt.plot(df['Close'])
plt.xlabel('Data', fontsize = '18')
plt.ylabel('Preco fechamento ', fontsize = '18')
plt.show()
data = df.filter(['Close'])
dataset = data.values
train_data_len = math.ceil(len(dataset)*0.8)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:train_data_len]
x_train = []
y_train = []
for i in range(60,len(train_data)): #tamanho da janela = 60
  x_train.append(train_data[i-60:i,0])#60 amostras, pos 0 a 59
  y_train.append(train_data[i-60:i,0])#01 e ocolocando na pos 60
  x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape
#construcao do modelo lstm
model = Sequential()
model.add(LSTM(50, return_sequences=True,input_shape =(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer ='adam', loss = 'mean_squared_error')

#treinamento
model.fit(x_train, y_train, batch_size = 1, epochs = 1)
test_data = scaled_data[train_data_len-60:,:]
x_test = []
y_test = []
y_test = dataset[train_data_len:,:]
for i in range(60,len(test_data)): #tamanho da janela = 60
  x_test.append(test_data[i-60:i,0])#60 amostras, pos 0 a 59
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
train = data [:train_data_len]
test = data[train_data_len:]

test ['Predictions'] = predictions
plt.figure(figsize = (16,8))
plt.title('Resultado da previsao')
plt.xlabel('Data', fontsize = '18')
plt.xlabel('Preco fechamento ', fontsize = '18')
plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.legend(['Treino', 'Teste','Previsao'],loc = 'lower right')
plt.show()