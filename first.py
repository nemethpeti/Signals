import numerapi
# a fork of yfinance that implements retries nicely
# pip install -e git+http://github.com/leonhma/yfinance.git@master#egg=yfinance
import yfinance
import simplejson

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests as re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, FR
from sklearn.ensemble import GradientBoostingRegressor

from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'svg'
import plotly
from functools import partial

napi = numerapi.SignalsAPI()
# In[download]

def load_data(download=False):
    

    full_data = pd.read_parquet('full_data.prq')
    
    last_date = full_data.index.max().date().strftime('%Y-%m-%d')
    print(last_date)

    # download
    if download == True:
        
        # read in yahoo to numerai ticker map, still a work in progress, h/t wsouza
        ticker_map = pd.read_csv('https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv')
        ticker_map.to_csv('signals_ticker_map_w_bbg.csv', index=False)
        print(f"Number of tickers in map: {len(ticker_map)}")
        
        # read in list of active Signals tickers which can change slightly era to era
        
        eligible_tickers = pd.Series(napi.ticker_universe(), name='numerai_ticker')
        print(f"Number of eligible tickers: {len(eligible_tickers)}")
        
        # map eligible numerai tickers to yahoo finance tickers
        yfinance_tickers = eligible_tickers.map(dict(zip(ticker_map['bloomberg_ticker'], ticker_map['yahoo']))).dropna()
        numerai_tickers = ticker_map['bloomberg_ticker']
        print(f'Number of eligible, mapped tickers: {len(yfinance_tickers)}')
        
        # download new data
        new_data = yfinance.download(yfinance_tickers.str.cat(sep=' '), start=last_date, threads=True)
        new_data = new_data['Adj Close'].stack().reset_index()
        
        # properly position and clean raw data, after taking adjusted close only
        new_data.columns = ['date', 'ticker', 'price']
        new_data.set_index('date', inplace=True)
        # convert yahoo finance tickers back to numerai tickers
        new_data['numerai_ticker'] = full_data.ticker.map(dict(zip(ticker_map['yahoo'], numerai_tickers)))
        
        full_data = pd.concat([full_data, new_data])
    
    print('Data downloaded.')
    print(f"Number of tickers with data: {len(full_data.numerai_ticker.unique())}")   
    
    return full_data

def RSI(df, interval=10):

    delta = df['price'].diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(interval).mean()
    RolDown = dDown.rolling(interval).mean().abs()

    RS = RolUp / RolDown
    RSI = 100.0 - (100.0 / (1.0 + RS))
    df[f'RSI{interval}'] = RSI
    return df

def quantile(df, source, target):
   
    df[target] = pd.qcut(df[source], 5, labels=False, duplicates='drop')
    return df

def add_lagged(df, source, lags):
    
    for lag in lags:
        df[f'{source}_lagged_{lag}'] = df[source].shift(lag)
        
    return df

    
def applyParallel(df, func, groupby, **kwargs):
    ticker_groups = df.groupby(groupby)
    fixed_func=partial(func, **kwargs)
    
    with Pool(cpu_count()) as p:
        ret_list = p.map(fixed_func, [group for name, group in ticker_groups])
        
    return pd.concat(ret_list)

def addDiffs(df, num_days, name='RSI_quintile_lag'):
    
    # create difference of the lagged features and absolute difference of the lagged features (change in RSI quintile by day)
    for day in range(num_days):
        df[f'{name}_diff_{day}'] = df[f'{name}_{day}'] - df[f'{name}_{day + 1}']
        df[f'{name}_abs_diff_{day}'] = np.abs( df[f'{name}_{day}'] - df[f'{name}_{day + 1}'])
        
    return df

def add_indicators(df):
    
    print('Calculating RSI')
    df = applyParallel(df, RSI, 'ticker', interval=10)
    #df = applyParallel(df, RSI, 'ticker', interval=50)
    
    print('Calculating Quantiles')
    df = applyParallel(df, quantile, df.index, source='RSI10', target='RSI10_quantile')
    df.dropna(inplace=True)
    
    print('Adding lags')
    df = applyParallel(df, add_lagged, 'ticker', source='RSI10_quantile', lags=range(6))

    print('Adding diffs')
    df = addDiffs(df, 5, 'RSI10_quantile_lagged')

    return df
 
def plotPriceSubplot(df, col):

    fig = plotly.subplots.make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df.index, y=df.price, name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col), row=2, col=1)

    fig.show()
    


# In[main]
if __name__ == '__main__':

    
    df = load_data()
    df = add_indicators(df)

# In[figures]

    #plotdf = df.loc[(df.ticker == 'AAPL') & (df.index > '2021-01-01')]
    #plotPriceSubplot(plotdf, 'RSI10')
    
    
    TARGET_NAME = 'target'
    PREDICTION_NAME = 'signal'
    
    try:
        targets = pd.read_csv('historical_targets.csv')
    except FileNotFoundError:
        napi.download_validation_data(dest_filename='historical_targets.csv')
        targets = pd.read_csv('historical_targets.csv')    
    
    targets['date'] = pd.to_datetime(targets['friday_date'], format='%Y%m%d')    
    targets.rename(columns={"bloomberg_ticker": "numerai_ticker"}, inplace=True)    
    
    # merge our feature data with Numerai targets
    data = pd.merge(df.reset_index(), targets, on=['date', 'numerai_ticker']).set_index('date')
    print(f'Number of eras in data: {len(data.index.unique())}')
        
    # for training and testing we want clean, complete data only
    data.dropna(inplace=True)
    data = data[data.index.weekday == 4]  # ensure we have only fridays
    data = data[data.index.value_counts() > 50]  # drop eras with under 50 observations per era
    data.to_parquet('ML_data.prq')
    
    # define column names of features, target, and prediction

    features = []
    features += [c for c in df if c.startswith("RSI10_qu")]
    features += [c for c in df if c.startswith("RSI10_diff")]
    features += [c for c in df if c.startswith("RSI10_abs_diff")]

    print(f'Features for training:\n {features}')
    

    # train test split
    train = data[data['data_type'] == 'train']
    val = data[data['data_type'] == 'validation']
    
    # train model
    print("Training model...")
    model = GradientBoostingRegressor(subsample=0.1)
    model.fit(train[features], train[TARGET_NAME])
    print("Model trained.")

    # predict test data
    val[PREDICTION_NAME] = model.predict(val[features])

    # predict live data
    # choose data as of most recent friday
    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    date_string = last_friday.strftime('%Y-%m-%d')
    
    try:
        live = df.loc[date_string].copy()
    except KeyError as e:
        print(f"No ticker on {e}")
        live = df.iloc[:0].copy()
    live.dropna(subset=features, inplace=True)

    # get data from the day before, for markets that were closed
    # on the most recent friday
    last_thursday = last_friday - timedelta(days=1)
    thursday_date_string = last_thursday.strftime('%Y-%m-%d')
    thursday_data = df.loc[thursday_date_string]
    # Only select tickers than aren't already present in live_data
    thursday_data = thursday_data[~thursday_data.ticker.isin(live.ticker.values)].copy()
    thursday_data.dropna(subset=features, inplace=True)
    
    live = pd.concat([live, thursday_data])
    
    print(f"Number of live tickers to submit: {len(live)}")
    live[PREDICTION_NAME] = model.predict(live[features])

    # prepare and writeout example file
    diagnostic_df = pd.concat([val, live])
    diagnostic_df['friday_date'] = diagnostic_df.friday_date.fillna(last_friday.strftime('%Y%m%d')).astype(int)
    diagnostic_df['data_type'] = diagnostic_df.data_type.fillna('live')
    diagnostic_df = diagnostic_df.sort_values(by=['ticker', 'friday_date'])
    diagnostic_df[['numerai_ticker', 'friday_date', 'data_type', 'signal']].reset_index(drop=True).to_csv('signal_upload.csv', index=False)
    print('Submission completed.')    
