import numpy as np 
import pandas as pd 
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
from keras.models import load_model
import streamlit as st 
import datetime
 

model = load_model('keras_model_new.h5')

# Title
st.title('Stock Trend Prediction')

#Start and End
start = '2010-01-01'
current_datetime=datetime.datetime.now()
# print("current date and time:",current_datetime)
end = current_datetime.strftime('%Y-%m-%d')
print(end)

#User Input
user_input = st.text_input("Enter Stock Ticker", 'AAPL')

#Making df
df = pdr.get_data_yahoo(user_input, start, end)
df = df.reset_index()
st.subheader('Data from 2010 - Present')
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100days MA')
fig = plt.figure(figsize = (12,6))
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100days MA & 200MA')
fig = plt.figure(figsize = (12,6))
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close)
st.pyplot(fig)

#Splitting into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =  pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#Scaling down Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)


#Model Making LSTM
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

#Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)

#scaling down inputs
inputs = scaler.transform(final_df)

x_test = [] 
y_test = []

#x-test and y-test
for i in range(100, (inputs.shape[0])):
    x_test.append(inputs[i-100: i])
    y_test.append(inputs[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Predictions
y_predicted = (model.predict(x_test)) 

#Scaling up values
scale = scaler.scale_[0]

y_predicted = scale * y_predicted
y_test = scale * y_test

st.subheader('Predictons vs Actual')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'r', label = 'Original Price')
plt.plot(y_predicted, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
