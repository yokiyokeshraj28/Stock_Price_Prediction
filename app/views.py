from urllib import request
from django.shortcuts import render

from plotly.offline import plot
import plotly.graph_objects as go
from plotly.graph_objs import Scatter
import json
import datetime as dt
from tensorflow import keras

from .models import Project


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import tensorflow as tf


# The Home page when Server loads up
def index(request):
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function
    data = yf.download(
        
        # passes the ticker
        tickers=['LT.NS', 'HDFCBANK.NS', 'ITC.NS', 'SBIN.NS', 'JSWSTEEL.NS', 'TATAMOTORS.NS'],

        group_by = 'ticker',
        
        threads=True, # Set thread value to true
        
        # used for access data[ticker]
        period='1y', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)



    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['LT.NS']['Adj Close'], name="LT")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['HDFCBANK.NS']['Adj Close'], name="HDFCBANK")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['ITC.NS']['Adj Close'], name="ITC")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['SBIN.NS']['Adj Close'], name="SBIN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['JSWSTEEL.NS']['Adj Close'], name="JSWSTEEL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['TATAMOTORS.NS']['Adj Close'], name="TATAMOTORS")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')
    


    # ================================================ To show recent stocks ==============================================
    
    df1 = yf.download(tickers = 'LT.NS', period='1d', interval='1d')
    df2 = yf.download(tickers = 'HDFCBANK.NS', period='1d', interval='1d')
    df3 = yf.download(tickers = 'ITC.NS', period='1d', interval='1d')
    df4 = yf.download(tickers = 'SBIN.NS', period='1d', interval='1d')
    df5 = yf.download(tickers = 'JSWSTEEL.NS', period='1d', interval='1d')
    df6 = yf.download(tickers = 'TATAMOTORS.NS', period='1d', interval='1d')

    df1.insert(0, "Ticker", "LT")
    df2.insert(0, "Ticker", "HDFCBANK")
    df3.insert(0, "Ticker", "ITC")
    df4.insert(0, "Ticker", "SBIN")
    df5.insert(0, "Ticker", "JSWSTEEL")
    df6.insert(0, "Ticker", "TATAMOTORS")

    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient ='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)

    # ========================================== Page Render section =====================================================

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })

def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value+'.NS', period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    Valid_Ticker = [
        "ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BPCL","BHARTIARTL","BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","HDFC","ICICIBANK","ITC","INDUSINDBK","INFY","JSWSTEEL","KOTAKBANK","LT","M&M","MARUTI","NTPC","NESTLEIND","ONGC","POWERGRID","RELIANCE","SBILIFE","SHREECEM","SBIN","SUNPHARMA","TCS","TATACONSUM","TATAMOTORS","TATASTEEL","TECHM","TITAN","UPL","ULTRACEMCO","WIPRO"
    ]

    if ticker_value not in Valid_Ticker:
        return render(request, 'Invalid_Ticker.html', {})
    
    if number_of_days < 0:
        return render(request, 'Negative_Days.html', {})
    
    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (INR per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')



    # ========================================== Machine Learning ==========================================

        
    start_date = str(dt.datetime.today() - dt.timedelta(days = 547.501))[:10]
    end_date = str(dt.datetime.today())[:10]
    top_page_df = yf.download(tickers = ticker_value+'.NS', start=start_date, end=end_date)

    # Convert dataset into suitable form to train the model

    train_end_date=str(dt.datetime.today() - dt.timedelta(days=152.083))[:10]
    test_start_date=str(dt.datetime.today() - dt.timedelta(days=121.667))[:10]

    train = top_page_df[:train_end_date]
    test = top_page_df[test_start_date:]

    def transform_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 7

    X_train, y_train = transform_dataset(train, train.Open, time_steps)
    X_test, y_test = transform_dataset(test, test.Open, time_steps)

    # Build the model

    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=200,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dense(units=1))
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.RMSprop()
    )

    # Training the model

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    # Forecasting using test data

    y_pred = model.predict(X_test)

    confidence= str(np.sqrt(np.mean(np.square(y_pred.flatten() - y_test)))/1000)

    print('RMSE:'+ confidence)

    no_of_days=number_of_days
    inputs = test[len(test) - (no_of_days): ].values

    pred_dict = {"date": [], "gp_pred": []}

    for i in range(0, no_of_days):
        pred_dict["date"].append(str(dt.datetime.today() + dt.timedelta(days=i))[0:10])

    for i in range(100, no_of_days):
        inputs = inputs.T
        inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
        pred_price = model.predict(inputs[:,i-152:i])
        inputs = np.append(inputs, pred_price)
        inputs = np.reshape(inputs, (inputs.shape[0], 1))

    for i in inputs:
        pred_dict["gp_pred"].append(i[0])
        
     # ========================================== Plotting predicted data ======================================
        
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['date'], y=pred_df['gp_pred'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Display Ticker Info ==========================================

    ticker = yf.Ticker(ticker_value+'.NS').info

    # ========================================== Page Render section ==========================================
    

    return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : confidence,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'Symbol':ticker['symbol'].rstrip('.NS'),
                                                    'Name':ticker['longName'],
                                                    'Current_Price':ticker['currentPrice'],
                                                    'Total_Revenue':ticker['totalRevenue'],
                                                    'Revenue_Per_Share':ticker['revenuePerShare'],
                                                    'Market_Cap':ticker['marketCap'],
                                                    'Country':ticker['country'],
                                                    'Quote_Type':ticker['quoteType'],
                                                    'Last_Dividend_Value':ticker['lastDividendValue'],
                                                    'Volume':ticker['volume'],
                                                    'Sector':ticker['sector'],
                                                    'Industry':ticker['industry']
                                                    })
