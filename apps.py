

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import pandas_datareader as data
import streamlit as st
import yfinance as yf
from keras.models import load_model

#start='2015-01-01'
#end='2019-12-31'

st.title('Stock Price Prediction')
#code_stock=st.text_input('label','')
user_input=st.text_input('Enter Stock Ticker','AAPL')
sta=st.text_input('Enter Stock date','2010-01-01')

df=yf.download(user_input,start=sta)
st.write(df)
df.dropna(inplace=True)

st.subheader('closing price')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler(feature_range=(0, 1))

#df.Close=scaler.fit_transform(df[['Close']])
df=df.Close


def sampling(sequence,n_steps):
  X,Y=list(),list()
  for i in range(len(sequence)):
    sample=i+n_steps
    if sample>len(sequence)-1:
      break
    x,y=sequence[i:sample],sequence[sample]
    X.append(x)
    Y.append(y)
  return np.array(X),np.array(Y)

n_steps=10
X,Y=sampling(df.to_list(),n_steps)


model=load_model('bit.h5')


plots=predicted=model.predict(X)
st.subheader('perfomence of predicting')
fig2=plt.figure(figsize=(12,6))
plt.plot(Y)
plt.plot(plots)
st.pyplot(fig2)



test=np.array(df.tail(10)).reshape(1,10)
predicted=model.predict(test)




test2=np.array(df[-12:-2]).reshape(1,10)
predicted2=model.predict(test2)

st.subheader("test values for next day")
st.write(predicted[0])

st.subheader("predicted values for today")
st.write(predicted2[0])







