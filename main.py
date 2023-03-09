import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)

# pd.set_option('display.max_rows', None)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_1 = pd.read_excel("C:\\Users\\User\Desktop\Ödev\datasets\online_retail_II.xlsx", sheet_name="Year 2009-2010")
df_2 = pd.read_excel("C:\\Users\\User\Desktop\Ödev\datasets\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df1 = df_1.copy()
df2 = df_2.copy()

#lets concat our datasets

df1["InvoiceDate"].max()
df2["InvoiceDate"].min()

#There some duplicated transactions

df2=df2[df2["InvoiceDate"] > df1["InvoiceDate"].max()]
#Now you can't

df2["InvoiceDate"].min()

#Lets concantrate them
df = pd.concat([df1 , df2], ignore_index=True)
df.info()

#EDA and Data Preparation

def check_df(data,head=5):
    print('#' * 40 + 'Shape' + '#' * 40)
    print(data.shape)
    print('#' * 40 + "NaN" + "#" * 40)
    print(data.isnull().sum())
    print('#'*40 + 'info' + '#'*40)
    print(data.info())
    print('#'*40 + "Describe" + '#'*40)
    print(data.describe().T)
    print('#' * 40 + "HEAD" + '#' * 40)
    print(data.head(head))

check_df(df)

#As you can see with te check_df function Quantity and Price has negative values

df = df[(df["Quantity"] > 0)]
df = df[df["Price"] > 0]

#And we don't need to use canceled Invoices

df = df[~df["Invoice"].str.contains("C", na=False)]
df.dropna(inplace=True)
df.describe().T

# Data have extreme values

def df_extreme(data=pd.DataFrame, variable=str):
    q1 = data[variable].quantile(0.01)
    q3 = data[variable].quantile(0.99)
    IQR = q3 - q1
    up_limit = q3 + 1.5 * IQR
    low_limit = q1 - 1.5 * IQR
    return low_limit , up_limit

def replace_extreme(data=pd.DataFrame, variable=str):
    low_limit, up_limit = df_extreme(data, variable)
    # data.loc[(data[variable] < low_limit)][variable] = low_limit
    data.loc[(data[variable] > up_limit),variable] = up_limit

replace_extreme(df, "Price")
replace_extreme(df, "Quantity")
df.describe().T
df.info()

#So we get rid of some dirt lets continue to calculate RFM metrics

df["total_price"] = df["Price"] * df["Quantity"]

df["InvoiceDate"].max()
case_date = dt.datetime(2011, 12, 11)
rfm = pd.DataFrame
rfm = df.groupby("Customer ID").agg({"Quantity": lambda x: x.count(),
                                     "total_price": lambda x: x.sum()})

rfm.head()
rfm.columns = ['total_transaction', 'total_price']

# Average Order Value (total_price/total_transaction)

rfm["AOV"] = rfm["total_price"]/rfm['total_transaction']

#Purchase Freq  (total_transaction/number of customers)

rfm["PurchFreq"] = rfm['total_transaction']/rfm.shape[0]

#Repeat Rate

repeat_rate = rfm[rfm["total_transaction"] > 1].shape[0]/rfm.shape[0]

ChurnRate = 1-repeat_rate

#Profit Margin %10

rfm["Profit_Margin"] = rfm["total_price"]/10

#Customer Value= AvgOrdVal*PurcFreq

rfm["Customer_Value"] = rfm["AOV"] * rfm["PurchFreq"]

#CLTV (CustVal/ChurnRate)*ProfMargn

rfm["CLV"] = ((rfm["total_price"]/rfm['total_transaction'] * rfm['total_transaction']/rfm.shape[0])/(1-(rfm[rfm["total_transaction"] > 1].shape[0]/rfm.shape[0]))) * (rfm["total_price"]/10)
rfm["CLV"] = (rfm["Customer_Value"]/ChurnRate)*rfm["Profit_Margin"]

#Segmentation
rfm.sort_values(by="CLV", ascending=False).head()
rfm.sort_values(by="CLV", ascending=False).tail()

rfm["segment"] = pd.qcut(rfm["CLV"], 4 , labels=["D", "C", "B", "A"])
rfm.groupby("segment").agg({"count", "mean", "sum"})

rfm.to_csv("cltv_c.csv")


