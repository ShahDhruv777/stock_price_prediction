import numpy as np 
import pandas as pd 
import pandas_datareader.data as pdr 
import matplotlib.pyplot as plt
import keras
import pickle
import datetime
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
scaler = MinMaxScaler()

#Model declaration
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (100, 1))) 
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True)) 
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True)) 
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu')) 
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.compile(optimizer='adam', loss='mean_squared_error')

str_arr=['AAPL','SBIN.NS','IBM','GOOG','TATAMOTORS.NS','TSLA','MRF.NS','ADANIENT.NS','RELIANCE.NS','HDB']
current_datetime=datetime.datetime.now()
print("current date and time:",current_datetime)
start='2000-04-01'
end = current_datetime.strftime('%Y-%m-%d')
# end='2015-05-01'
print(end)

# df = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

for companies in str_arr:
  df = pdr.get_data_yahoo(companies, start, end)
  df = df.reset_index()
  df = df.drop(['Adj Close', 'Date'], axis=1)
  data_training = pd.DataFrame(df['Close'][0:int(len(df))])
  # data_testing =  pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

  print(data_training.shape)
  # print(data_testing.shape)
  data_training_array = scaler.fit_transform(data_training)
  x_train = []
  y_train = []

  for i in range(100, data_training_array.shape[0]):
      x_train.append(data_training_array[i-100: i])
      y_train.append(data_training_array[i,0])
  x_train, y_train = np.array(x_train), np.array(y_train)
  model.fit(x_train, y_train, epochs = 50,validation_split=0.2,batch_size=4,callbacks=[callback])

model.save('keras_model.h5')

