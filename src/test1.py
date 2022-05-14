import pandas as pd

df=pd.read_csv('/home/yuriy/Personal/CODE/salesForecast/data/finalDataframe.csv')

df['family'] = df['family'].str.replace(' ', '').str.replace('/','').str.replace(',','')

df['storeCat']=df['family'].astype(str)+'_'+df['store_nbr'].astype(str)

# https://stackoverflow.com/questions/22798934/pandas-long-to-wide-reshape-by-two-variables
# https://stackoverflow.com/questions/35414625/pandas-how-to-run-a-pivot-with-a-multi-index
df=df[['onpromotion', 'dom', 'dow', 'wom', 'month',
       'isMonthStart', 'isMonthEnd', 'wagePaid', 'daysFromLastStipend',
       'transactions', 'isHoliday', 'oil_price', 'sales', 'storeCat']]


pivot_test=df.pivot_table(index=['dom', 'dow', 'wom', 'month'], columns=['storeCat'], values=['sales','onpromotion','isMonthStart', 'isMonthEnd', 'wagePaid', 'daysFromLastStipend',
'transactions', 'isHoliday', 'oil_price'])
