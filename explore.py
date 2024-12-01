from rich import print as rprint

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from plotnine.ggplot import ggplot
from plotnine import (
    aes,
    facet_wrap,
    geom_point,
    geom_hline,
    geom_line,
    geom_boxplot,
    ggtitle,
    labs,
)


# === Loading data ===
his_data = pd.read_excel(io="2024 Historische day-ahead-Preise Strom DE.xlsx")

his_data.head()

his_data.shape

# Deal with missing values
his_data.interpolate(method="linear", inplace=True)

# Deal with outliers
# fig_rawdata = (
#     ggplot(his_data, mapping=aes("Datum", "Preis"))
#     + geom_line()
#     # + geom_hline(yintercept=0)
#     + ggtitle("Hourly prices")
#     + labs(x="Time", y="Prices")
# )
# fig_rawdata.draw(show=True)

# Outliers
outlier_df = his_data.copy()

outlier_df.insert(loc=0, column="return", value=0.0)

Preis = outlier_df["Preis"].copy()
len_ = len(Preis)
Preis_ = Preis.iloc[1:len_].reset_index(drop=True)
Preis_l1 = Preis.iloc[0 : (len_ - 1)].reset_index(drop=True)

return_ = (Preis_ - Preis_l1) / Preis_l1

outlier_df.loc[1:len_, "return"] = return_

perc = 1
lower_threshold = h_df.groupby("year")["nh_preis"].quantile(q=perc / 100)
upper_threshold = h_df.groupby("year")["nh_preis"].quantile(q=1 - perc / 100)

outlier_df = h_df.merge(
    right=lower_threshold, on="year", suffixes=("", "_lower")
)
outlier_df = outlier_df.merge(
    right=upper_threshold, on="year", suffixes=("", "_upper")
)
outlier_df.head()
outlier_df.columns

fig_outliers = (
    ggplot(outlier_df)
    + geom_line(mapping=aes("Datum", "nh_preis"))
    + geom_line(mapping=aes("Datum", "nh_preis_lower"), color="red")
    + geom_line(mapping=aes("Datum", "nh_preis_upper"), color="red")
    + ggtitle("Hourly Prices: Detecting Outliers")
    + labs(x="Time", y="Prices")
)
fig_outliers.draw(show=True)
fig_outliers.save("hi.pdf")

# Remove outliers
down_spike = outlier_df["nh_preis"] < outlier_df["nh_preis_lower"]
up_spike = outlier_df["nh_preis"] > outlier_df["nh_preis_upper"]

wo_outlier_df = outlier_df.copy()

wo_outlier_df.loc[down_spike, "nh_preis"] = np.nan
wo_outlier_df.loc[up_spike, "nh_preis"] = np.nan

wo_outlier_df.interpolate(method="linear", inplace=True)

fig_wo_outliers = (
    ggplot(wo_outlier_df, mapping=aes("Datum", "nh_preis"))
    + geom_line()
    + ggtitle("Hourly Prices: After Removing Outliers")
    + labs(x="Time", y="Prices")
)
fig_wo_outliers.draw(show=True)


# === Transformations ===
# Split the date
his_data.insert(loc=1, column="hour", value=-1000)
his_data.insert(loc=1, column="day", value=-1000)
his_data.insert(loc=1, column="month", value=-1000)
his_data.insert(loc=1, column="year", value=-1000)

for ii in range(his_data.shape[0]):
    his_data.loc[ii, "hour"] = his_data.loc[ii, "Datum"].hour
    his_data.loc[ii, "day"] = his_data.loc[ii, "Datum"].day
    his_data.loc[ii, "month"] = his_data.loc[ii, "Datum"].month
    his_data.loc[ii, "year"] = his_data.loc[ii, "Datum"].year

his_data.head()

# Hourly prices
h_df = his_data.copy()

h_df.rename(columns={"Preis": "h_preis"}, inplace=True)

h_df.head()

# Daily prices
d_df = h_df.groupby(["year", "month", "day"], as_index=False)["h_preis"].mean()

d_df.rename(columns={"h_preis": "d_preis"}, inplace=True)

d_df.insert(loc=0, column="date", value=np.nan)
d_df["date"] = pd.to_datetime(d_df[["year", "month", "day"]])

d_df.head()

# Yearly prices
y_df = d_df.groupby("year", as_index=False)["d_preis"].mean()

y_df.rename(columns={"d_preis": "y_preis"}, inplace=True)

y_df.head()

# Normalize the prices
d_df = d_df.merge(right=y_df[["year", "y_preis"]], on="year")

d_df["nd_preis"] = d_df["d_preis"] / d_df["y_preis"]

d_df.head()


h_df = h_df.merge(
    right=d_df[["year", "month", "day", "nd_preis"]],
    on=["year", "month", "day"],
)

h_df["nh_preis"] = h_df["h_preis"] / h_df["nd_preis"]

h_df.head()

# # === Build the clusters ===
# his_data.insert(loc=0, column="weekday", value=np.nan)
# for ii in range(his_data.shape[0]):
#     his_data.loc[ii, "weekday"] = his_data.loc[ii, "Datum"].weekday()


# # === Data exploration ===
# # Hourly price
# fig_h_preis = (
#     ggplot(h_df, mapping=aes("Datum", "h_preis"))
#     + geom_line()
#     # + geom_hline(yintercept=0)
#     + facet_wrap("year", scales="free")
#     + ggtitle("Hourly prices")
#     + xlab("Time")
#     + ylab("Prices")
# )

# fig_h_preis.draw(show=True)

# # Normalized hourly prices
# fig_nh_preis = (
#     ggplot(h_df[h_df["year"] < 2023], mapping=aes(x="Datum", y="nh_preis"))
#     + geom_line()
#     + facet_wrap("year", scales="free")
#     + ggtitle("Normalized hourly prices")
#     + labs(x="Time", y="Prices")
# )

# fig_nh_preis.draw(show=True)

# # Daily prices
# fig_d_preis = (
#     ggplot(d_df[d_df["year"] < 2023], mapping=aes(x="date", y="d_preis"))
#     + geom_line()
#     + facet_wrap("year", scales="free")
#     + ggtitle("Daily prices")
#     + labs(x="Time", y="Prices")
# )

# fig_d_preis.draw(show=True)


# # Normalized daily prices
# fig_nd_preis = (
#     ggplot(d_df[d_df["year"] < 2023], mapping=aes(x="date", y="nd_preis"))
#     + geom_line()
#     + facet_wrap("year", scales="free")
#     + ggtitle("Daily prices")
#     + labs(x="Time", y="Prices")
# )

# fig_nd_preis.draw(show=True)


# # Yearly prices
# fig_y_preis = (
#     ggplot(y_df, mapping=aes(x="year", y="y_preis"))
#     + geom_line()
#     + ggtitle("Yearly Prices")
#     + labs(x="Year", y="Price")
# )
# fig_y_preis.draw(show=True)


# # January norm. daily prices
# for ii in range(1, 13):
#     fig_mon_preis = (
#         ggplot(d_df[d_df["month"] == ii], mapping=aes(x="day", y="nd_preis"))
#         + geom_line()
#         + facet_wrap("year", scales="free")
#         + ggtitle(f"Month {ii}")
#         + labs(x="Time", y="Prices")
#     )

#     fig_mon_preis.draw(show=True)


# Box plot for exploring seasonality
fig = (
    ggplot(d_df, mapping=aes(x="factor(month)", y="nd_preis"))
    + geom_boxplot()
    + ggtitle("Daily prices")
    + labs(x="Time", y="Prices")
)
fig.draw(show=True)


# === Seasonality dummies for a year ===
d_df.insert(loc=4, column="weekday", value=-1000)

for ii in range(d_df.shape[0]):
    d_df.loc[ii, "weekday"] = d_df.loc[ii, "date"].weekday()

d_df.tail()

d_df_dummies = pd.get_dummies(
    d_df,
    prefix=["weekday", "month"],
    drop_first=True,
    columns=["weekday", "month"],
)
d_df_dummies.head()
d_df_dummies.tail()


# X_mat = pd.concat([d_df, month_dummies], axis=1)
X_mat = d_df_dummies.loc[:, "weekday_1":"month_12"]
X_mat.head()
X_mat.tail()

Y_vec = d_df["nd_preis"]
Y_vec.head()
Y_vec.tail()

reg_year_shape = LinearRegression().fit(X=X_mat, y=Y_vec)

reg_year_shape.score(X=X_mat, y=Y_vec)

# Seasonality dummies for for a day
h_df_dummies = pd.get_dummies(
    h_df, prefix="hour", drop_first=True, columns=["hour"]
)
h_df_dummies.head()
h_df_dummies.tail()

h_df.insert(loc=5, column="weekday", value=-1000)

for ii in range(h_df.shape[0]):
    h_df.loc[ii, "weekday"] = h_df.loc[ii, "Datum"].weekday()

h_df.head()
h_df.tail()

month_names = np.array(
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
    ]
)

for ii in range(11):
    month_name = month_names[ii]

    name_wd = "wd_" + month_name
    name_sat = "sat_" + month_name
    name_sun = "sun_" + month_name

    h_df_dummies[name_wd] = (h_df["month"] == ii) * (
        h_df["weekday"].between(0, 4)
    )
    h_df_dummies[name_sat] = (h_df["month"] == ii) * (h_df["weekday"] == 5)
    h_df_dummies[name_sun] = (h_df["month"] == ii) * (h_df["weekday"] == 6)

X_mat_day = h_df_dummies.loc[:, "hour_1":"sun_Nov"]
Y_vec_day = h_df_dummies.loc[:, "nh_preis"]

reg_day_shape = LinearRegression().fit(X=X_mat_day, y=Y_vec_day)
reg_day_shape.score(X=X_mat_day, y=Y_vec_day)

# First tests
# Download from eex
# Check on the eex data

# === Autocorrelation before and after de-seasonalization

# === MAE, RMSE ===


# === Backtesting ===
