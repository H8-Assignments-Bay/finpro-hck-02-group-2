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
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup

def news(selected_stock):
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
    
  df_news = pd.DataFrame()

  # Loop through each heading element and print the text
  for heading_element in heading_elements:
    heading = heading_element.get_text()

    # create a temporary DataFrame with one column and one row
    temp_df_news = pd.DataFrame({'title': [heading]})
  
    # append the temporary DataFrame to the original DataFrame
    df_news = df_news.append(temp_df_news, ignore_index=True)




  # Create an empty list to store the links
  links = []

  # Loop through each link element and store the link
  for link_element in link_elements:
    link = link_element.get('href')
    links.append(link)

  # Create a dataframe from the links
  df_links = pd.DataFrame(links, columns=['Link'])


  df_news = pd.concat([df_news, df_links], axis=1)
  #st.write(df_news.head(5))
  x = df_news.title
  st.metric(label="x", value=123, delta=123, delta_color="off")

  return df_news

# def headline(selected_stock):
#   google_news = GNews(country='Indonesia')
#   if selected_stock == 'BBRI':
#     news = google_news.get_news('BRI saham')
#     #values=["bank mandiri","BBRI"]
#   elif selected_stock == 'BBCA':
#     news = google_news.get_news('BBCA saham')
#     #values=["bank mandiri","BBCA"]
#   elif selected_stock == 'BBNI':
#     news = google_news.get_news('BNI')
#     #values=["bank mandiri","BNI"]
#   elif selected_stock == 'BBTN':
#     news = google_news.get_news('BTN saham')
#     #values=["bank mandiri","BTN"]
#   else:
#     news = google_news.get_news('BMRI saham')
#     # values=["bank mandiri","BMRI"]
#   df = pd.DataFrame(news)
#   df['published date'] = pd.to_datetime(df['published date'])
#   df.sort_values(by='published date', ascending=False, inplace=True)
#   #df = df.loc[df['title'].isin(values)]
#   #df = df[df['title'].str.contains(selected_stock, regex=False)]
#   df.reset_index(drop=True, inplace=True)
#   df.head(10)
#   st.write(df.head(5))
#   return df

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
# Load the Model
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

st.title('Stock Forecast App')

stocks = ('BBCA', 'BMRI', 'BBNI', 'BBRI','BBTN')
selected_stock = st.sidebar.selectbox('Stocks', stocks)

n_years = st.sidebar.slider('Days of prediction:', 1, 7)

st.subheader('Historical Stock Report')
st.write(selected_stock)
# Set the ticker
if selected_stock == 'BBCA':
    ticker = 'BBCA.JK'
    data = yf.download(ticker, START, TODAY)
    data = data[["Open", "High","Low", "Close","Volume"]]
    st.write(data.tail())

    data.reset_index(inplace=True)
    data =data.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                    'Adj Close':'adj_close','Volume':'volume'})

    data['date'] = pd.to_datetime(data.date)
    data = data['close']

    input_data = scaler_bca.transform(np.array(data).reshape(-1,1))

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 8
    X_test, y_test = create_dataset(input_data, time_step)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)    


    y_predicted = model_bca.predict(X_test)

    y_predicted = scaler_bca.inverse_transform(y_predicted)
    y_test = scaler_bca.inverse_transform(y_test.reshape(-1,1))
    st.subheader("Predictions vs Original")
    fig2= plt.figure(figsize = (12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    news(selected_stock)

elif selected_stock == 'BMRI':
    ticker = 'BMRI.JK'
    data = yf.download(ticker, START, TODAY)
    data = data[["Open", "High","Low", "Close","Volume"]]
    st.write(data.tail())

    data.reset_index(inplace=True)
    data =data.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                    'Adj Close':'adj_close','Volume':'volume'})

    data['date'] = pd.to_datetime(data.date)
    data = data['close']

    input_data = scaler_bmri.transform(np.array(data).reshape(-1,1))

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 8
    X_test, y_test = create_dataset(input_data, time_step)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)    


    y_predicted = model_bmri.predict(X_test)

    y_predicted = scaler_bmri.inverse_transform(y_predicted)
    y_test = scaler_bmri.inverse_transform(y_test.reshape(-1,1))
    st.subheader("Predictions vs Original")
    fig2= plt.figure(figsize = (12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    news(selected_stock)

elif selected_stock == 'BBNI':
    ticker = 'BBNI.JK'
    data = yf.download(ticker, START, TODAY)
    data = data[["Open", "High","Low", "Close","Volume"]]
    st.write(data.tail())

    data.reset_index(inplace=True)
    data =data.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                    'Adj Close':'adj_close','Volume':'volume'})

    data['date'] = pd.to_datetime(data.date)
    data = data['close']

    input_data = scaler_bni.transform(np.array(data).reshape(-1,1))

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 8
    X_test, y_test = create_dataset(input_data, time_step)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)    


    y_predicted = model_bni.predict(X_test)

    y_predicted = scaler_bni.inverse_transform(y_predicted)
    y_test = scaler_bni.inverse_transform(y_test.reshape(-1,1))
    st.subheader("Predictions vs Original")
    fig2= plt.figure(figsize = (12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    news(selected_stock)

elif selected_stock == 'BBRI':
    ticker = 'BBRI.JK'
    data = yf.download(ticker, START, TODAY)
    data = data[["Open", "High","Low", "Close","Volume"]]
    st.write(data.tail())

    data.reset_index(inplace=True)
    data =data.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                    'Adj Close':'adj_close','Volume':'volume'})

    data['date'] = pd.to_datetime(data.date)
    data = data['close']

    input_data = scaler_bri.transform(np.array(data).reshape(-1,1))

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 8
    X_test, y_test = create_dataset(input_data, time_step)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)    


    y_predicted = model_bri.predict(X_test)

    y_predicted = scaler_bri.inverse_transform(y_predicted)
    y_test = scaler_bri.inverse_transform(y_test.reshape(-1,1))
    st.subheader("Predictions vs Original")
    fig2= plt.figure(figsize = (12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    news(selected_stock)

elif selected_stock == 'BBTN':
    ticker = 'BBTN.JK'
    data = yf.download(ticker, START, TODAY)
    data = data[["Open", "High","Low", "Close","Volume"]]
    st.write(data.tail())

    data.reset_index(inplace=True)
    data =data.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                    'Adj Close':'adj_close','Volume':'volume'})

    data['date'] = pd.to_datetime(data.date)
    data = data['close']

    input_data = scaler_btn.transform(np.array(data).reshape(-1,1))

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 8
    X_test, y_test = create_dataset(input_data, time_step)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)    


    y_predicted = model_btn.predict(X_test)

    y_predicted = scaler_btn.inverse_transform(y_predicted)
    y_test = scaler_btn.inverse_transform(y_test.reshape(-1,1))
    st.subheader("Predictions vs Original")
    fig2= plt.figure(figsize = (12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
    news(selected_stock)


