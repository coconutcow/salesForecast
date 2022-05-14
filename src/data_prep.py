import pandas as pd

# Reading data src files
df_train=pd.read_csv(r'C:\San\Projects\Python\salesForecast\data\train.csv\train.csv')
df_train['date'] = pd.to_datetime(df_train['date'])
df_transactions=pd.read_csv(r'C:\San\Projects\Python\salesForecast\data\transactions.csv\transactions.csv')
df_transactions['date'] = pd.to_datetime(df_transactions['date'])
df_holidays=pd.read_csv(r'C:\San\Projects\Python\salesForecast\data\holidays_events.csv')
df_holidays['date'] = pd.to_datetime(df_holidays['date'])
df_oil=pd.read_csv(r'C:\San\Projects\Python\salesForecast\data\oil.csv')
df_oil['date'] = pd.to_datetime(df_oil['date'])

# Train file
# basic date time values extracted
df_train['dom'] = df_train['date'].dt.day
df_train['dow'] = df_train['date'].dt.day_name()
df_train['wom'] = df_train['date'].apply(lambda d: (d.day-1) // 7 + 1)
df_train['month'] = df_train['date'].dt.month
df_train['year'] = df_train['date'].dt.year
df_train['isMonthStart']=df_train['date'].dt.is_month_start
df_train['isMonthEnd']=df_train['date'].dt.is_month_end
df_train["wagePaid"] = df_train[["dom", "isMonthEnd"]].apply(lambda x: True if (x.dom == 15) else(True if (x.isMonthEnd == True) else False),axis=1)
df_train['daysFromLastStipend']=df_train['dom'].apply(lambda x: x if (x<15) else (x-15))
df_train['daysFromLastStipend']=df_train['dom'].apply(lambda x: x if (x<15) else (x-15))

# Merge df_train w/ transactions
df=df_train.merge(df_transactions,on=['date','store_nbr'],how='left').fillna(0)

#
df_holidays=df_holidays[df_holidays['transferred']!=True].reset_index(drop=True)
df_holidays['type']=True

print(df_holidays)

df=df.merge(df_holidays[['date','type']],on=['date'],how='left',suffixes="isHoliday_").fillna(False)

df.columns = ['id', 'date', 'store_nbr', 'family','sales','onpromotion','dom','dow','wom','month','year','isMonthStart','isMonthEnd','wagePaid','daysFromLastStipend','transactions','isHoliday']

r = pd.date_range(start=df_oil.date.min(), end=df_oil.date.max())
df_oil=df_oil.set_index("date").reindex(r).ffill().bfill().reset_index()
df_oil.columns=['date','oil_price']
df=df.merge(df_oil,on=['date'],how='left').fillna(0)

df=df[['store_nbr', 'family', 'onpromotion', 'dom',
       'dow', 'wom', 'month', 'isMonthStart', 'isMonthEnd', 'wagePaid',
       'daysFromLastStipend', 'transactions', 'isHoliday', 'oil_price','sales']]
df.to_csv('finalDataframe.csv',index=False)
