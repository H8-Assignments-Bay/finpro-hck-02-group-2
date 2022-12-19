import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gnews import GNews
from newspaper import Article
import datetime
import requests
from bs4 import BeautifulSoup

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

def news(selected_stock):
  st.header("Headline News")
  if selected_stock == 'BBRI':
    url = "https://www.google.com/search?sxsrf=ALiCzsaW9Ve40dpjT2L2MbzLZSyj0UfoYA:1671421151309&q=bbri+saham&tbm=nws&source=univ&tbo=u&sxsrf=ALiCzsaW9Ve40dpjT2L2MbzLZSyj0UfoYA:1671421151309&sa=X&ved=2ahUKEwixqvn_4IT8AhW5TWwGHbb0B_AQt8YBKAB6BAgXEAE&biw=1536&bih=714&dpr=1.25"
  elif selected_stock == 'BBCA':
    url = "https://www.google.com/search?q=bbca+saham&biw=1536&bih=714&tbm=nws&sxsrf=ALiCzsYZF65FYfcDywaK6tqFhgQoX_4C2Q%3A1671435753828&ei=6RWgY7qVMt2n3LUPlKOVmAI&ved=0ahUKEwj6rfyyl4X8AhXdE7cAHZRRBSMQ4dUDCA0&uact=5&oq=bbca+saham&gs_lcp=Cgxnd3Mtd2l6LW5ld3MQAzIECAAQQzIFCAAQgAQyBQgAEIAEMgQIABBDMgQIABBDMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIGCAAQBxAeOgoIABCxAxCDARBDOgYIABAWEB46CAgAEAgQBxAeOgcIABCxAxBDOgsIABCABBCxAxCDAToKCAAQgAQQsQMQCjoHCAAQgAQQCjoHCAAQgAQQDVCjHFjsJWDSK2gAcAB4AIAB0gGIAf8HkgEFNS4zLjGYAQCgAQHAAQE&sclient=gws-wiz-news"
  elif selected_stock == 'BBNI':
    url = "https://www.google.com/search?q=bbni+saham&biw=1536&bih=714&tbm=nws&sxsrf=ALiCzsaT9AndHtEm-mKIFpFeUriuDnPI5A%3A1671435851611&ei=SxagY8_qJPPnz7sPhM2uGA&ved=0ahUKEwiPvMzhl4X8AhXz83MBHYSmCwMQ4dUDCA0&uact=5&oq=bbni+saham&gs_lcp=Cgxnd3Mtd2l6LW5ld3MQAzILCAAQgAQQsQMQgwEyBQgAEIAEMgYIABAHEB4yBggAEAcQHjIFCAAQgAQyBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB4yBAgAEB46DQgAEIAEELEDEIMBEAo6BwgAEIAEEAo6CAgAEAcQHhAKUKgMWKgMYPgQaAFwAHgAgAFNiAHLAZIBATOYAQCgAQHAAQE&sclient=gws-wiz-news"
  elif selected_stock == 'BBTN':
    url = "https://www.google.com/search?q=bbtn+saham&biw=1536&bih=714&tbm=nws&sxsrf=ALiCzsZJPwVQ0WJor-EFo7dY-IB9FaryuQ%3A1671435863728&ei=VxagY6DSK46A3LUPtvihmAQ&ved=0ahUKEwjg2a_nl4X8AhUOALcAHTZ8CEMQ4dUDCA0&uact=5&oq=bbtn+saham&gs_lcp=Cgxnd3Mtd2l6LW5ld3MQAzINCAAQgAQQsQMQgwEQDTIECAAQQzIGCAAQBxAeMgQIABAeMgQIABAeMgQIABAeMgYIABAFEB4yBggAEAgQHjIGCAAQCBAeMgYIABAIEB46CggAELEDEIMBEEM6CwgAEIAEELEDEIMBOgUIABCABDoICAAQCBAHEB46CggAEIAEELEDEAo6BwgAEIAEEAo6BwgAEIAEEA06CAgAEAcQHhAKOgoIABAIEAcQHhAKOg8IABCABBCxAxCDARANEApQ5wZYkhRgkBZoAnAAeACAAXSIAYwGkgEDNi4zmAEAoAEBwAEB&sclient=gws-wiz-news"
  else:
    url = "https://www.google.com/search?tbm=nws&sxsrf=ALiCzsZtLZUC7bGTbjg_NdM6EDljm8dkBQ:1671435919925&q=bmri+saham&spell=1&sa=X&ved=2ahUKEwjuiZaCmIX8AhXKnNgFHXUYBQgQBSgAegQIBxAB&biw=1536&bih=714&dpr=1.25"

  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
  page = requests.get(url, headers=headers)
  soup = BeautifulSoup(page.content, "lxml")

  # Find all heading elements
  heading_elements = soup.find_all('div', class_='mCBkyc ynAwRc MBeuO nDgy9d')

  link_elements = soup.find_all('a', class_='WlydOe')

  # Loop through each link element and store the link


  for link_element in link_elements:
    link = link_element.get('href')
    headline =  link_element.get_text()
    st.markdown(f'<p class="big-font">{headline}</p>', unsafe_allow_html=True)
    st.markdown(link, unsafe_allow_html=True)
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

# Load the models and scalers
scaler_bca = pickle.load(open('scaler_bca.pkl','rb'))
model_bca = tf.keras.models.load_model('model_bca.h5')
scaler_bmri = pickle.load(open('scaler_bmri.pkl','rb'))
model_bmri = tf.keras.models.load_model('model_bmri.h5')
scaler_bni = pickle.load(open('scaler_bni.pkl','rb'))
model_bni = tf.keras.models.load_model('model_bni.h5')
scaler_bri = pickle.load(open('scaler_bri.pkl','rb'))
model_bri = tf.keras.models.load_model('model_bri.h5')
scaler_btn = pickle.load(open('scaler_btn.pkl','rb'))
model_btn = tf.keras.models.load_model('model_btn.h5')

# Dictionary to map stock names to ticker symbols and models/scalers
stock_data = {
    'BBCA': {'ticker': 'BBCA.JK', 'model': model_bca, 'scaler': scaler_bca},
    'BMRI': {'ticker': 'BMRI.JK', 'model': model_bmri, 'scaler': scaler_bmri},
    'BBNI': {'ticker': 'BBNI.JK', 'model': model_bni, 'scaler': scaler_bni},
    'BBRI': {'ticker': 'BBRI.JK', 'model': model_bri, 'scaler': scaler_bri},
    'BBTN': {'ticker': 'BBTN.JK', 'model': model_btn, 'scaler': scaler_btn},
}

# Set the start and end dates for the historical data
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set the time step for the model
time_step = 8

# Set up the web app
st.title('Stock Forecast App')

# Allow the user to select a stock and specify the number of days to predict
stocks = ('BBCA', 'BMRI', 'BBNI', 'BBRI','BBTN')
selected_stock = st.sidebar.selectbox('Stocks', stocks)
n_days = st.sidebar.slider('Days of prediction:', 7, 30)

# Display the historical data for the selected stock
st.subheader('Historical Stock Report')
st.write(selected_stock)

# Get the data for the selected stock
data = yf.download(stock_data[selected_stock]['ticker'], START, TODAY)
data = data[["Open", "High","Low", "Close","Volume"]]
st.write(data.tail())

# Preprocess the data
data.reset_index(inplace=True)
data = data.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                            'Adj Close':'adj_close','Volume':'volume'})
data['date'] = pd.to_datetime(data.date)
data = data['close']
input_data = stock_data[selected_stock]['scaler'].transform(np.array(data).reshape(-1,1))

# Create the dataset for the model
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Split the data into test and train sets
X_test, y_test = create_dataset(input_data, time_step)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

# Make predictions using the model
y_predicted = stock_data[selected_stock]['model'].predict(X_test)

# Inverse transform the predictions and original data
y_predicted = stock_data[selected_stock]['scaler'].inverse_transform(y_predicted)
y_test = stock_data[selected_stock]['scaler'].inverse_transform(y_test.reshape(-1,1))

# Plot the predictions vs the original data
st.subheader("Predictions vs Original")
fig2= plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Display the prediction for the specified number of days
st.subheader("Prediction for the next {} days".format(n_days))
fig3= plt.figure(figsize = (12,6))
plt.plot(y_predicted[-n_days:], 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
news(selected_stock)
