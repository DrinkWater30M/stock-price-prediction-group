import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import external_stock_data
import prediction
from datetime import date


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))



df_nse = pd.read_csv("./NSE-TATA.csv")

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']


data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

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

model=load_model("model/saved_model.h5")

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



df= pd.read_csv("./NSE-TATA.csv")

# implement ui
app.layout = html.Div([
   
   # title
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    # tool bar
    html.Div(
        style={"display": "flex", "gap": "20px"},
        children=[
            dcc.Dropdown(
                id='coin-dropdown',
                options=['BTC-USD', 'ETH-USD', 'ADA-USD'], 
                value='BTC-USD', 
                clearable=False,
                style={"width": "200px"}),
            dcc.Dropdown(
                id='price-type-dropdown',
                options=[
                    {'label': 'Open Price', 'value': 'Open'},
                    {'label': 'Close Price', 'value': 'Close'},
                    {'label': 'Low Price', 'value': 'Low'},
                    {'label': 'High Price', 'value': 'High'},
                ], 
                value='Open', 
                clearable=False,
                style={"width": "200px"}),
    ]),

    # present data by graph
    html.Div(
        children = [
            html.H2("Actual And Predicted Closing Prices(LSTM)",style={"textAlign": "center"}),
            dcc.Graph(id="price-graph"),
            html.H2("Transactions Volume",style={"textAlign": "center"}),
            dcc.Graph(id="volume-graph")				
	    ],
        style={"border": "solid 1px gray", "marginTop": "10px"}  
    ),



    dcc.Tabs(id="tabs", children=[
        # example stock
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data(Example)',children=[
			html.Div([
				html.H2("Actual And Predicted Closing Prices(LSTM)",style={"textAlign": "center"}),
				dcc.Graph(
					id="price",
					# figure={
					# 	"data":[
					# 		go.Scatter(
					# 			x=valid.index,
					# 			y=valid["Close"],
					# 			mode='markers'
					# 		)

					# 	],
					# 	"layout":go.Layout(
					# 		title='scatter plot',
					# 		xaxis={'title':'Date'},
					# 		yaxis={'title':'Closing Rate'}
					# 	)
					# },
                    figure = {
                        'data': [
                            go.Scatter(
                                x=new_data.index,
								y=new_data["Close"],
								mode='lines',
                                opacity=0.7, 
                                name=f'Actual closing price',textposition='bottom center'),
                            go.Scatter(
                                x=valid.index,
								y=valid["Predictions"],
								mode='lines',
                                opacity=0.6,
                                name=f'Predicted closing price',textposition='bottom center')

                        ],
                        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                                        '#FF7400', '#FFF400', '#FF0056'],
                        height=600,
                        yaxis={"title":"Price (USD)"})
                    },                  
                
				),
				
				html.H2("Transactions Volume",style={"textAlign": "center"}),
                dcc.Graph(
					id="volume",
                    figure = {
                        'data': [
                            go.Scatter(
                                x=new_data.index,
								y=data["Total Trade Quantity"],
                                mode='lines', opacity=0.7,
                                name=f'Volume', textposition='bottom center')
                        ], 
                        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                                        '#FF7400', '#FFF400', '#FF0056'],
                        height=600,
                        yaxis={"title":"Volume"})
                    }
				),				
			])      
        ]),

        # btc-usd stock
        dcc.Tab(
            value='btc-usd',
            label='BTC-USD Stock Data',
            children=[
			html.Div([
				html.H2("Actual And Predicted Closing Prices(LSTM)",style={"textAlign": "center"}),
				dcc.Graph(id="btc-usd-price"),
				html.H2("Transactions Volume",style={"textAlign": "center"}),
				dcc.Graph(id="btc-usd-volume")				
			])   
        ]),

        # eth-usd stock
        dcc.Tab(
            value='eth-usd',
            label='ETH-USD Stock Data',
            children=[
			html.Div([
				html.H2("Actual And Predicted Closing Prices(LSTM)",style={"textAlign": "center"}),
				dcc.Graph(id="eth-usd-price"),
				html.H2("Transactions Volume",style={"textAlign": "center"}),
				dcc.Graph(id="eth-usd-volume")				
			])   
        ]),

        # # info some example stock
        # dcc.Tab(label='Facebook Stock Data', children=[
        #     html.Div([
        #         html.H1("Stocks High vs Lows", 
        #                 style={'textAlign': 'center'}),
              
        #         dcc.Dropdown(id='my-dropdown',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple','value': 'AAPL'}, 
        #                               {'label': 'Facebook', 'value': 'FB'}, 
        #                               {'label': 'Microsoft','value': 'MSFT'}], 
        #                      multi=True,value=['FB'],
        #                      style={"display": "block", "margin-left": "auto", 
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='highlow'),
        #         html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
        #         dcc.Dropdown(id='my-dropdown2',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple','value': 'AAPL'}, 
        #                               {'label': 'Facebook', 'value': 'FB'},
        #                               {'label': 'Microsoft','value': 'MSFT'}], 
        #                      multi=True,value=['FB'],
        #                      style={"display": "block", "margin-left": "auto", 
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='volume')
        #     ], className="container"),
        # ])
    ])
])


# update btc graph when click tab
@app.callback(Output('btc-usd-price', 'figure'),
              [Input('tabs', 'value')])
def update_price_graph(valueTab):
     if valueTab == 'btc-usd':
        predColumn = 'Close'
        predPrice = prediction.predictByLSTM('BTC-USD', predColumn, '2023-01-01', date.today())
        dataPrice = external_stock_data.getStockDataToNow('BTC-USD', 5*365)
        figure = {
            'data': [
                go.Scatter(
                    x=dataPrice.index,
                    y=dataPrice[predColumn],
                    mode='lines',
                    opacity=0.7, 
                    name=f'Actual closing price',textposition='bottom center'),
                go.Scatter(
                    x=predPrice.index,
                    y=predPrice["Predictions"],
                    mode='lines',
                    opacity=0.6,
                    name=f'Predicted closing price',textposition='bottom center')
            ],
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            yaxis={"title":"Price (USD)"})
        }
        return figure
     
     return {}

@app.callback(Output('btc-usd-volume', 'figure'),
              [Input('tabs', 'value')])
def update_volume_graph(valueTab):
     if valueTab == 'btc-usd':
        dataVolume = external_stock_data.getStockDataToNow('BTC-USD', 5*365)
        figure = {
            'data': [
                go.Scatter(
                    x=dataVolume.index,
                    y=dataVolume["Volume"],
                    mode='lines', opacity=0.7,
                    name=f'Volume', textposition='bottom center')
            ], 
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            yaxis={"title":"Volume"})
        }
        return figure
     
     return {}

# update eth graph when click tab
@app.callback(Output('eth-usd-price', 'figure'),
              [Input('tabs', 'value')])
def update_price_graph(valueTab):
     if valueTab == 'eth-usd':
        predColumn = 'Close'
        predPrice = prediction.predictByLSTM('ETH-USD', predColumn, '2023-01-01', date.today())
        dataPrice = external_stock_data.getStockDataToNow('ETH-USD', 5*365)
        figure = {
            'data': [
                go.Scatter(
                    x=dataPrice.index,
                    y=dataPrice[predColumn],
                    mode='lines',
                    opacity=0.7, 
                    name=f'Actual closing price',textposition='bottom center'),
                go.Scatter(
                    x=predPrice.index,
                    y=predPrice["Predictions"],
                    mode='lines',
                    opacity=0.6,
                    name=f'Predicted closing price',textposition='bottom center')
            ],
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            yaxis={"title":"Price (USD)"})
        }
        return figure
     
     return {}

@app.callback(Output('eth-usd-volume', 'figure'),
              [Input('tabs', 'value')])
def update_volume_graph(valueTab):
     if valueTab == 'eth-usd':
        dataVolume = external_stock_data.getStockDataToNow('ETH-USD', 5*365)
        figure = {
            'data': [
                go.Scatter(
                    x=dataVolume.index,
                    y=dataVolume["Volume"],
                    mode='lines', opacity=0.7,
                    name=f'Volume', textposition='bottom center')
            ], 
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            yaxis={"title":"Volume"})
        }
        return figure
     
     return {}



# update price graph follow by input user
@app.callback(Output('price-graph', 'figure'),
              [
                  Input('coin-dropdown', 'value'),
                  Input('price-type-dropdown', 'value')
              ])
def update_price_graph(coin, price_type):
    predPrice = prediction.predictByLSTM(coin, price_type, '2023-01-01', date.today())
    dataPrice = external_stock_data.getStockDataToNow(coin, 5*365)
    figure = {
        'data': [
            go.Scatter(
                x=dataPrice.index,
                y=dataPrice[price_type],
                mode='lines',
                opacity=0.7, 
                name=f'Actual {price_type} Price',textposition='bottom center'),
            go.Scatter(
                x=predPrice.index,
                y=predPrice["Predictions"],
                mode='lines',
                opacity=0.6,
                name=f'Predicted {price_type} Price',textposition='bottom center')
        ],
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                        '#FF7400', '#FFF400', '#FF0056'],
        height=600,
        yaxis={"title":"Price (USD)"})
    }
    return figure

# update volume graph follow by input user
@app.callback(Output('volume-graph', 'figure'),
              [
                  Input('coin-dropdown', 'value'),
              ])
def update_volume_graph(coin):
    dataVolume = external_stock_data.getStockDataToNow(coin, 5*365)
    figure = {
        'data': [
            go.Scatter(
                x=dataVolume.index,
                y=dataVolume["Volume"],
                mode='lines', opacity=0.7,
                name=f'Volume', textposition='bottom center')
        ], 
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                        '#FF7400', '#FFF400', '#FF0056'],
        height=600,
        yaxis={"title":"Volume"})
    }
    return figure

# #update graph for stock info tab
# @app.callback(Output('highlow', 'figure'),
#               [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown):
#     dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
#     trace1 = []
#     trace2 = []
#     for stock in selected_dropdown:
#         trace1.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["High"],
#                      mode='lines', opacity=0.7, 
#                      name=f'High {dropdown[stock]}',textposition='bottom center'))
#         trace2.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["Low"],
#                      mode='lines', opacity=0.6,
#                      name=f'Low {dropdown[stock]}',textposition='bottom center'))
#     traces = [trace1, trace2]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
#                                             '#FF7400', '#FFF400', '#FF0056'],
#             height=600,
#             title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
#             xaxis={"title":"Date",
#                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'count': 6, 'label': '6M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'step': 'all'}])},
#                    'rangeslider': {'visible': True}, 'type': 'date'},
#              yaxis={"title":"Price (USD)"})}
#     return figure


# @app.callback(Output('volume', 'figure'),
#               [Input('my-dropdown2', 'value')])
# def update_graph(selected_dropdown_value):
#     dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
#     trace1 = []
#     for stock in selected_dropdown_value:
#         trace1.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["Volume"],
#                      mode='lines', opacity=0.7,
#                      name=f'Volume {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data, 
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
#                                             '#FF7400', '#FFF400', '#FF0056'],
#             height=600,
#             title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
#             xaxis={"title":"Date",
#                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'count': 6, 'label': '6M',
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'step': 'all'}])},
#                    'rangeslider': {'visible': True}, 'type': 'date'},
#              yaxis={"title":"Transactions Volume"})}
#     return figure



if __name__=='__main__':
	app.run_server(debug=True)