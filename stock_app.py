import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import external_stock_data
import prediction
from datetime import date

# init app
app = dash.Dash()
server = app.server

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
            dcc.Dropdown(
                id='algorithm-dropdown',
                options=[
                    {'label': 'LSTM', 'value': 'lstm'},
                    {'label': 'RNN', 'value': 'rnn'},
                    {'label': 'XGBoost', 'value': 'xgboost'},
                ], 
                value='lstm', 
                clearable=False,
                style={"width": "200px"}),
    ]),

    # data presention by graph
    html.Div(
        children = [
            html.H2("Actual And Predicted Prices(LSTM)",style={"textAlign": "center"}),
            dcc.Loading(
                dcc.Graph(id="price-graph"),
            ),

            html.H2("Transactions Volume",style={"textAlign": "center"}),
            dcc.Loading(
                dcc.Graph(id="volume-graph")				
            ),
        ],
        style={"border": "solid 1px gray", "marginTop": "10px"}  
    ),
])

# update price graph follow by input user
@app.callback(Output('price-graph', 'figure'),
              [
                  Input('coin-dropdown', 'value'),
                  Input('price-type-dropdown', 'value'),
                  Input('algorithm-dropdown', 'value')
              ])
def update_price_graph(coin, price_type, algorithm):
    if algorithm == 'rnn':
        print('runing RNN algorithm')
        predPrice = prediction.predictByRNN(coin, price_type, '2023-01-01', date.today())
    elif algorithm == 'xgboost':
        print('runing XGBoost algorithm')
        predPrice = prediction.predictByXGBoost(coin, price_type, '2023-01-01', date.today())
    else:
        print('runing default algorithm: LSTM')
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
    print('run')
    dataVolume = external_stock_data.getStockData(coin, '2018-01-01', '2023-01-01')
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

# start app
if __name__=='__main__':
	app.run_server(debug=True)