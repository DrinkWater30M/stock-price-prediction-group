import yfinance as yf
ticker = yf.Ticker('GOOGL').info
previous_close_price = ticker['regularMarketPreviousClose']
print('Ticker: GOOGL')
print('Previous Close Price:', previous_close_price)