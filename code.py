#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as py
import requests
import datetime as dt
import time
import talib
from talib import abstract
import quantstats

py.style.use('ggplot')
pd.set_option('display.max_rows', None)

df1 = pd.read_csv('TWF_Futures_Minute_Trade.txt')


df1.index = pd.to_datetime(df1['Date'] + ' ' + df1['Time'])
df1 = df1.drop(columns=['Date','Time'])
print(df1.tail())
df2 = pd.read_csv('TXF_outsample.csv')
df2['TotalVolume'] = [0] * len(df2)
df2.columns.values[0] = "Datetime"
df2.index = pd.to_datetime(df2['Datetime'])
df2 = df2.drop('Datetime', axis='columns')
print(df2.tail())
print(df2.shape)
df = pd.concat([df1, df2])
#print(df.tail())
df.columns = ['open', 'high', 'low', 'close', 'volume']
df['Hour'] = df.index.map(lambda x: x.hour)
df.head(3)

rule = '3T'
Morning = df[(df['Hour'] >= 8) & (df['Hour'] <= 13)]
Morning.index = Morning.index + dt.timedelta(minutes=15)
Morning.iloc[0:8]

# 確認 9:03 及 9:06 的 open 是 9:01 及 9:04 的 open
Morning.resample(rule=rule, closed='right', label='right').first()[['open']].iloc[0:3]

# label='left' => 時間轉為左側時間
Morning.resample(rule=rule, closed='right', label='left').first()[['open']].iloc[0:3]

rule = '60T'

Morning = df[(df['Hour'] >= 8) & (df['Hour'] <= 13)]
Morning.index = Morning.index + dt.timedelta(minutes=15)

d1 = Morning.resample(rule=rule, closed='right', label='left').first()[['open']]
d2 = Morning.resample(rule=rule, closed='right', label='left').max()[['high']]
d3 = Morning.resample(rule=rule, closed='right', label='left').min()[['low']]
d4 = Morning.resample(rule=rule, closed='right', label='left').last()[['close']]
d5 = Morning.resample(rule=rule, closed='right', label='left').sum()[['volume']]

df_Morning = pd.concat([d1,d2,d3,d4,d5], axis=1)
df_Morning = df_Morning.dropna()
df_Morning.index = df_Morning.index - dt.timedelta(minutes=15)
df_Morning.head()

rule = '60T'

Night = df[(df['Hour'] < 8) | (df['Hour'] > 13)]

d1 = Night.resample(rule=rule, closed='right', label='left').first()[['open']]
d2 = Night.resample(rule=rule, closed='right', label='left').max()[['high']]
d3 = Night.resample(rule=rule, closed='right', label='left').min()[['low']]
d4 = Night.resample(rule=rule, closed='right', label='left').last()[['close']]
d5 = Night.resample(rule=rule, closed='right', label='left').sum()[['volume']]

df_Night = pd.concat([d1,d2,d3,d4,d5], axis=1)
df_Night = df_Night.dropna()
df_Night.head()

df_Day = pd.concat([df_Morning, df_Night], axis=0)
df_Day = df_Day.sort_index(ascending=True)
df_Day.head()

df_Morning['Hour'] = df_Morning.index.map(lambda x: x.hour)

trainData = df_Morning[(df_Morning.index >= '2010-01-01 00:00:00') & (df_Morning.index <= '2023-12-31 00:00:00')].copy()
testData = df_Morning[(df_Morning.index >= '2010-1-1 00:00:00') & (df_Morning.index <= '2019-5-22 00:00:00')].copy()
testData.head(3)

settlementDate_ = pd.read_csv('settlementDate.csv', encoding = 'utf-8')
settlementDate_.columns = ['settlementDate', 'futures', 'settlementPrice']
settlementDate_

bool_ = [False if 'W' in i else True for i in settlementDate_['futures']]

settlementDate = [i.replace('/','-') for i in list(settlementDate_[bool_]['settlementDate'])]

settlementDate = [pd.to_datetime(i).date() for i in settlementDate]

#參數設定
fund = 1000000
feePaid = 600
lowerRSI = 10
upperRSI = 80
RSIlength = 7
MAlength = 40
NumStd = 0.5
K = 0.04

#指標計算
# trainData['H'] = trainData['high'].shift(1).rolling(window=length, center=False).max()
# trainData['L'] = trainData['low'].shift(1).rolling(window=length, center=False).min()
trainData['MA'] = trainData['close'].rolling(window=MAlength, center=False).mean()
trainData['STD'] = trainData['close'].rolling(window=MAlength, center=False).std()
trainData['upLine'] = trainData['MA'] + NumStd*trainData['STD']
trainData['downLine'] = trainData['MA'] - NumStd*trainData['STD']
trainData['rsi'] = abstract.RSI(trainData.close, RSIlength)
print(trainData.tail(3))


df_arr = np.array(trainData)
time_arr = np.array(trainData.index)
date_arr = [pd.to_datetime(i).date() for i in time_arr]

#建立存放資料的單位
BS = None
buy = []
sell = []
sellshort = []
buytocover = []
profit_list = [0]
buy_profit = [0]
sell_profit = [0]
profit_fee_list = [0]
fee_buy = [0]
fee_sell = [0]
profit_fee_list_realized = []
rets = []

for i in range(len(df_arr)):

    if i == len(df_arr)-1:
        break

    ## 進場邏輯
    entryLong = df_arr[i,3] > df_arr[i,8] and df_arr[i, 10] >= upperRSI
    entrySellShort = df_arr[i,3] < df_arr[i,9] and df_arr[i, 10] <= lowerRSI
    entryCondition = date_arr[i] not in settlementDate

    ## 出場邏輯
    exitShort = df_arr[i,3] <= df_arr[i,6]
    exitBuyToCover = df_arr[i,3] >= df_arr[i,6]
    exitCondition = date_arr[i] in settlementDate and df_arr[i,5] >= 11

    ## 停利停損邏輯
    if BS == 'B':
        stopLoss = df_arr[i,3] <= df_arr[t,0] * (1-K)
        stopProfit = df_arr[i,3] >= df_arr[t,0] * (1+K)
    elif BS == 'S':
        stopLoss = df_arr[i,3] >= df_arr[t,0] * (1+K)
        stopProfit = df_arr[i,3] <= df_arr[t,0] * (1-K)
    stopLoss = False
    stopProfit = False

#     if exitCondition == True:
#         print(f'{time_arr[i]}')

    if BS == None:
        profit_list.append(0)
        buy_profit.append(0)
        sell_profit.append(0)
        profit_fee_list.append(0)
        fee_buy.append(0)
        fee_sell.append(0)

        if entryLong and entryCondition:
            BS = 'B'
            t = i+1
            buy.append(t)
            print("Enter Long Position")
            print("Buy Price: {}, time: {}".format(df_arr[t,0], time_arr[t]))

        elif entrySellShort and entryCondition:
            BS = 'S'
            t = i+1
            sellshort.append(t)
            print("Enter Short Position")
            print("Sell Price: {}, time: {}".format(df_arr[t,0], time_arr[t]))

    elif BS == 'B':
        profit = 200 * (df_arr[i+1,0] - df_arr[i,0])
        profit_list.append(profit)
        buy_profit.append(profit)
        sell_profit.append(0)

        if exitShort or i == len(df_arr)-2 or exitCondition or stopLoss or stopProfit:
            pl_round = 200 * (df_arr[i+1,0] - df_arr[t,0])
            profit_fee = profit - feePaid*2
            profit_fee_list.append(profit_fee)
            fee_buy.append(profit_fee)
            fee_sell.append(0)
            sell.append(i+1)
            BS=None
            print("Sell Price: {}, time: {}".format(df_arr[i+1,0], time_arr[i+1]))
            print("Trade completed")
            print()

            # Realized PnL
            profit_fee_realized = pl_round - feePaid*2
            profit_fee_list_realized.append(profit_fee_realized)
            rets.append(profit_fee_realized/(200*df_arr[t,0]))

        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)
            fee_buy.append(profit_fee)
            fee_sell.append(0)

    elif BS == 'S':
        profit = 200 * (df_arr[i,0] - df_arr[i+1,0])
        profit_list.append(profit)
        sell_profit.append(profit)
        buy_profit.append(0)

        if exitBuyToCover or i == len(df_arr)-2 or exitCondition or stopLoss or stopProfit:
            pl_round = 200 * (df_arr[t,0] - df_arr[i+1,0])
            profit_fee = profit - feePaid*2
            profit_fee_list.append(profit_fee)
            fee_sell.append(profit_fee)
            fee_buy.append(0)
            buytocover.append(i+1)
            BS = None
            print("Buycover Price: {}, time: {}".format(df_arr[i+1,0], time_arr[i+1]))
            print("Trade completed")
            print()

            # Realized PnL
            profit_fee_realized = pl_round - feePaid*2
            profit_fee_list_realized.append(profit_fee_realized)
            rets.append(profit_fee_realized/(200*df_arr[t,0]))

        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)
            fee_sell.append(profit_fee)
            fee_buy.append(0)


equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=trainData.index)

equity_buy = pd.DataFrame({'profit':np.cumsum(buy_profit), 'profitfee':np.cumsum(fee_buy)}, index=trainData.index)
equity_sell = pd.DataFrame({'profit':np.cumsum(sell_profit), 'profitfee':np.cumsum(fee_sell)}, index=trainData.index)
# print(equity)
equity.plot(y = 'profit', grid=True, figsize=(12,6));
equity_buy.plot(y = 'profit', grid=True, figsize=(12,6));
equity_sell.plot(y = 'profit', grid=True, figsize=(12,6));

equity['equity'] = equity['profitfee'] + fund
equity_buy['equity'] = equity_buy['profitfee'] + fund
equity_sell['equity'] = equity_sell['profitfee'] + fund

equity['drawdown_percent'] = (equity['equity']/equity['equity'].cummax()) - 1
equity_buy['drawdown_percent'] = (equity_buy['equity']/equity_buy['equity'].cummax()) - 1
equity_sell['drawdown_percent'] = (equity_sell['equity']/equity_sell['equity'].cummax()) - 1

equity['drawdown'] = equity['equity'] - equity['equity'].cummax() #前n個元素的最大值
equity_buy['drawdown'] = equity_buy['equity'] - equity_buy['equity'].cummax() #前n個元素的最大值
equity_sell['drawdown'] = equity_sell['equity'] - equity_sell['equity'].cummax() #前n個元素的最大值
#print(profit_fee_list_realized)

profit = equity['profitfee'].iloc[-1]
profit_buy = equity_buy['profitfee'].iloc[-1]
profit_sell = equity_sell['profitfee'].iloc[-1]

ret = equity['equity'][-1]/equity['equity'][0] - 1
ret_long = equity_buy['equity'][-1]/equity_buy['equity'][0] - 1
ret_short = equity_sell['equity'][-1]/equity_sell['equity'][0] - 1

mdd = abs(equity['drawdown_percent'].min())
mdd_long = abs(equity_buy['drawdown_percent'].min())
mdd_short = abs(equity_sell['drawdown_percent'].min())

calmarRatio = ret / mdd
tradeTimes = len(buy) + len(sellshort)
winRate = len([i for i in profit_fee_list_realized if i > 0]) / len(profit_fee_list_realized)
profitFactor = sum([i for i in profit_fee_list_realized if i>0]) / abs(sum([i for i in profit_fee_list_realized if i<0]))

print('Profit : ',profit)
print('Return : ',ret)
print('Max DrawDown : ',mdd)
print('Caimar Ratio : ',calmarRatio)
print('Trade Times : ',tradeTimes)
print('Win Rate : ',winRate)
print('Profit Factor : ',profitFactor)
print('buytimes', len(buy))
print('selltimes', len(sellshort))
print('profit long:', profit_buy)
print('profit short:', profit_sell)
print('long return:', ret_long)
print('short return:', ret_short)
print('long mdd:', mdd_long)
print('short mdd:', mdd_short)

print(equity.head())
equity.plot(y='equity', figsize=(12, 6))

# 時間損益(年)
equity.index = pd.to_datetime(equity.index) #確保索引是datetime型態
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
#years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
#years = ['2020', '2021', '2022', '2023']
year_ret = []
year_mdd = []
print(equity.index.year.unique())
for i in equity.index.year.unique():
    year_ret.append(equity['equity'][str(i)].iloc[-1] / equity['equity'][str(i)].iloc[0] - 1)
    drawdown_percent = (equity['equity'][str(i)]/equity['equity'][str(i)].cummax()) - 1
    year_mdd.append(abs(drawdown_percent.min()))
df = pd.DataFrame({'Return':year_ret},index = years)
df_mdd = pd.DataFrame({'MDD':year_mdd},index = years)
# heatmap函式
py.figure(figsize=(10,1))
sns.heatmap(df.transpose(), annot=True, cmap='OrRd')
py.title('Return by year')
py.show()
print('')

py.figure(figsize=(10,1))
sns.heatmap(df_mdd.transpose(), annot=True, cmap='OrRd')
py.title('MDD by year')
py.show()
print('')


ret = equity['equity'].pct_change(periods=1).dropna()
ret_buy = equity_buy['equity'].pct_change(periods=1).dropna()
ret_sell = equity_sell['equity'].pct_change(periods=1).dropna()

#quantstats.reports.full(ret)
quantstats.reports.full(ret, '^TWII')
#quantstats.reports.full(ret_buy)
#quantstats.reports.full(ret_sell)
