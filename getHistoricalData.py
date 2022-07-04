import pandas as pd
from datetime import datetime

def helper(client, symbol, interval, fromDate, toDate) -> pd:
    klines = client.get_historical_klines(symbol, interval, fromDate, toDate)
    df = pd.DataFrame(klines, columns=['dateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    df.dateTime = pd.to_datetime(df.dateTime, unit='ms')
    df['Date'] = df.dateTime.dt.strftime("%d/%m/%Y")
    df['Time'] = df.dateTime.dt.strftime("%H:%M:%S")
    df = df.drop(['dateTime', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol','takerBuyQuoteVol', 'ignore'], axis=1)
    column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    df = df.reindex(columns=column_names)
    return df

def fetch(Client, client, symbol, startDate, endDate, csvName) -> pd:
    fromDate = str(datetime.strptime(startDate, '%d/%m/%Y'))
    toDate = str(datetime.strptime(endDate, '%d/%m/%Y')) 
    sym = symbol
    interval = Client.KLINE_INTERVAL_1DAY
    df = helper(client, sym, interval, fromDate, toDate)
    fileName = csvName
    df.to_csv(fileName + '.csv')
    return df