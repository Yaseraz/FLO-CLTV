##############################################################
# Flo CLTV Prediction
##############################################################


############################################################################
# Preparing the data
############################################################################

#Libraries
import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


# Import Dataset
df_=pd.read_csv("PycharmProjects/scientificProject/flo_data_20k.csv", sep=",")
df=df_.copy()

# create a function called check_df to examine the data in general.
def check_df(dataframe, shape=True, columns=True, info=True, na=True, desc=True):
    """
    This function checks DataFrame's Descriptions

    :param dataframe: A pandas dataframe
    :param shape: checks Dataframe's Shape
    :param columns: Prints Columns Names
    :param info: Prints Columns types
    :param na: Count and prints NaN values in DataFrame
    :param desc: Prints DataFrame's Descriptions
            like min, max, first, second, third quantile, Variance and Mean values
    :return: Data frame's Shape,
            Names of DataFrame's columns, DataFrame's columns Type,
            how many NA values in DataFrame, Descriptive Stats
    """

    outputs = []
    if shape:
        outputs.append(('Shape', dataframe.shape))
    if columns:
        outputs.append(('Columns', dataframe.columns))
    if info:
        outputs.append(('Types', dataframe.dtypes))
    if na:
        outputs.append(('NA', dataframe.isnull().sum()))
    if desc:
        outputs.append(('Descriptive', dataframe.describe().T))
    for output in outputs:
        print(15 * "#", output[0], 15 * "#")
        print(output[1], "\n")

check_df(df)


# Creating two functions for suppressing outliers
def outlier_thresholds(dataframe,variable):
    Q1=dataframe[variable].quantile(0.01)
    Q3=dataframe[variable].quantile(0.99)
    IQR=Q3-Q1
    low_limit=Q1-IQR*1.5
    up_limit=Q3+IQR*1.5
    return low_limit,up_limit

# Note: When calculating cltv(freq.,t,monetary,recency) values must be int so Low limit and up limit must be round
def replace_with_thresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit.round()
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit.round()

# Set the list (integer variables) for suppressing outliers

l=["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]

for i in l:
    replace_with_thresholds(df,i)

# Create new variables for each one customer's spending and purchasing
df["total_order"]=df["order_num_total_ever_offline"] + df["order_num_total_ever_offline"]
df["total_value"]=df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


############################################################################
# Creating CLTV Data Structure
############################################################################

# Changing the type of variable as a date
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns]=df[date_columns].apply(pd.to_datetime)
df[date_columns].info()

# Determine the analysis date.
today_date=dt.datetime(2021,6,1)

# Create Cltv
cltv=pd.DataFrame()
cltv["CustomerId"]=df["master_id"]
cltv["recency"] = ((df["last_order_date"]-df["first_order_date"]).astype("timedelta64[D]"))/7
cltv["T"]=((today_date-df["first_order_date"]).astype("timedelta64[D]"))/7
cltv["frequency"]=df["total_order"]
cltv["monetary"]=df["total_value"]/df["total_order"]
cltv.head()

############################################################################
# Establishing the BG/NBD,gamma-gamma model and calculating Cltv
############################################################################

# BG-NBD Model

bgf=BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv["frequency"],cltv["recency"],cltv["T"])


cltv["exp_sales_3_month"] = bgf.predict(4*3,
            cltv["frequency"],
            cltv["recency"],
            cltv["T"])

cltv["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv["frequency"],
                                       cltv["recency"],
                                       cltv["T"])


#Gamma-Gamma Model
ggf=GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv["frequency"],cltv["monetary"])


cltv["expected_average_profit"]=ggf.conditional_expected_average_profit(cltv["frequency"],cltv["monetary"])

cltv["exp_profit_6month"]=ggf.conditional_expected_average_profit(cltv["frequency"],cltv["monetary"])

cltv["cltv"]= ggf.customer_lifetime_value(bgf,
                                 cltv["frequency"],
                                 cltv["recency"],
                                 cltv["T"],
                                 cltv["monetary"],
                                 time=6,
                                 freq="M",
                                 discount_rate=0.01)

cltv["cltv"].sort_values(ascending=False).head(20)



cltv["segment"]=pd.qcut(cltv["cltv"],4,labels=["D","C","B","A"])

new_group = cltv.loc[(cltv["segment"]=="A") | (cltv["segment"] == "B")]

new_group.to_csv("A and B customers")

############################################################################
# Task 5: Functionalizing the whole process
############################################################################

def outlier_thresholds(dataframe,variable):
    Q1=dataframe[variable].quantile(0.01)
    Q3=dataframe[variable].quantile(0.99)
    IQR=Q3-Q1
    low_limit=Q1-IQR*1.5
    up_limit=Q3+IQR*1.5
    return low_limit,up_limit


def replace_with_thresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit.round()
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit.round()


def create_cltv(dataframe,csv=False):
    # Data prep
    l = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
         "customer_value_total_ever_online"]
    for i in l:
       replace_with_thresholds(df, i)

    df["total_order"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_offline"]
    df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

    date_columns = df.columns[df.columns.str.contains("date")]
    df[date_columns] = df[date_columns].apply(pd.to_datetime)
    today_date = dt.datetime(2021, 6, 1)

    # Creating CLTV Data Structure
    cltv = pd.DataFrame()
    cltv["CustomerId"] = df["master_id"]
    cltv["recency"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
    cltv["T"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
    cltv["frequency"] = df["total_order"]
    cltv["monetary"] = df["total_value"] / df["total_order"]

    # BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv["frequency"], cltv["recency"], cltv["T"])

    cltv["exp_sales_3_month"] = bgf.predict(4 * 3,
                                            cltv["frequency"],
                                            cltv["recency"],
                                            cltv["T"])

    cltv["exp_sales_6_month"] = bgf.predict(4 * 6,
                                            cltv["frequency"],
                                            cltv["recency"],
                                            cltv["T"])

    # Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)

    ggf.fit(cltv["frequency"], cltv["monetary"])

    cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv["frequency"], cltv["monetary"])

    cltv["exp_profit_6month"] = ggf.conditional_expected_average_profit(cltv["frequency"], cltv["monetary"])

    cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                               cltv["frequency"],
                                               cltv["recency"],
                                               cltv["T"],
                                               cltv["monetary"],
                                               time=6,
                                               freq="M",
                                               discount_rate=0.01)
    # Create segment by Cltv
    cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])
    new_group = cltv.loc[(cltv["segment"] == "A") | (cltv["segment"] == "B")]

    if csv:
        new_group.to_csv("A  and B customers")

    return cltv,new_group


df=df_.copy()

create_cltv(df)
