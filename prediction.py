# Imports:
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# To run plt in jupyter or gg colab envirionment
# %matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras.models import load_model
import external_stock_data
from datetime import date, timedelta



from sklearn.preprocessing import MinMaxScaler

def predictByLSTM(stock, column, start_date, end_date):
    # # get data
    # df = external_stock_data.getStockData(stock, start_date, end_date)
    # df["Date"] = df.index

    # data=df.sort_index(ascending=True,axis=0)
    # new_data=pd.DataFrame(index=range(0,len(df)),columns=['Date', column])

    # for i in range(0,len(data)):
    #     new_data["Date"][i]=data['Date'][i]
    #     new_data[column][i]=data[column][i]

    # new_data.index=new_data.Date
    # new_data.drop("Date",axis=1,inplace=True)

    # scaler=MinMaxScaler(feature_range=(0,1))
    # nSections = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    # inputs=new_data[len(new_data)-nSections:].values
    # inputs=inputs.reshape(-1,1)
    # new_inputs=scaler.transform(inputs)

    # X_test=[]
    # for i in range(nSections,new_inputs.shape[0]):
    #     X_test.append(new_inputs[i-nSections:i,0])
    # X_test=np.array(X_test)
    # X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    
    # model=load_model("BTC-USD_lstm_model.h5")
    # predict_price=model.predict(X_test)
    # predict_price=scaler.inverse_transform(predict_price)

    # new_data['Predictions']=predict_price
    scaler=MinMaxScaler(feature_range=(0,1))



    df = external_stock_data.getStockData(stock, start_date, end_date)
    df["Date"] = df.index


    data=df.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data["Close"][i]=data["Close"][i]

    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)

    dataset=new_data.values

    train=dataset[0:987,:]
    valid=dataset[987:,:]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    x_train,y_train=[],[]

    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
        
    x_train,y_train=np.array(x_train),np.array(y_train)

    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    model=load_model("BTC-USD_lstm_model.h5")

    inputs=new_data[len(new_data)-len(valid)-60:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)

    X_test=[]
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)

    train=new_data[:987]
    valid=new_data[987:]
    valid['Predictions']=closing_price
    return valid