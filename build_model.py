# 1. Imports:
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# To run plt in jupyter or gg colab envirionment
# %matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
import external_stock_data

def buildModelByLSTM(stock, column, days):
    # get data
    df = external_stock_data.getStockDataToNow(stock, days)

    # 3. Analyze the closing prices from dataframe:
    df["Date"] = df.index

    # 4. Sort the dataset on date time and filter “Date” and “Close” columns:
    data=df.sort_index(ascending=True,axis=0)
    new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date', column])

    for i in range(0,len(data)):
        new_dataset["Date"][i]=data['Date'][i]
        new_dataset[column][i]=data[column][i]

    # 5. Normalize the new filtered dataset:
    # get close price column
    new_dataset.index=new_dataset.Date
    new_dataset.drop("Date",axis=1,inplace=True)
    final_dataset=new_dataset.values

    # get range to train data and valid data
    train_data=final_dataset

    # scale close price to range 0,1
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(final_dataset)

    x_train_data,y_train_data=[],[]

    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-60:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

    x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    # 6. Build and train the LSTM model:
    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error',optimizer='adam')
    lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

    # 8. Save the LSTM model:
    lstm_model.save(f"{stock}_{column}_lstm_model.h5")
    return

# build some model
##### BTC-USD #####
buildModelByLSTM('BTC-USD', "Open", 5*365)
buildModelByLSTM('BTC-USD', "Close", 5*365)
buildModelByLSTM('BTC-USD', "Low", 5*365)
buildModelByLSTM('BTC-USD', "High", 5*365)

##### ETH-USD #####
buildModelByLSTM('ETH-USD', "Open", 5*365)
buildModelByLSTM('ETH-USD', "Close", 5*365)
buildModelByLSTM('ETH-USD', "Low", 5*365)
buildModelByLSTM('ETH-USD', "High", 5*365)

##### ADA-USD #####
buildModelByLSTM('ADA-USD', "Open", 5*365)
buildModelByLSTM('ADA-USD', "Close", 5*365)
buildModelByLSTM('ADA-USD', "Low", 5*365)
buildModelByLSTM('ADA-USD', "High", 5*365)