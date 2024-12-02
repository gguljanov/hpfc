from rich import print as rprint

import numpy as np
import pandas as pd

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


# === Missing Values ===
his_data.interpolate(method="linear", inplace=True)


# === Detect Outliers ===
fig_rawdata = (
    ggplot(his_data, mapping=aes("Datum", "Preis"))
    + geom_line()
    # + geom_hline(yintercept=0)
    + ggtitle("Hourly prices")
    + labs(x="Time", y="Prices")
)
fig_rawdata.draw(show=True)

# Outliers
outlier_df = his_data.copy()

outlier_df.insert(loc=1, column="year", value=-1000)

for ii in range(outlier_df.shape[0]):
    outlier_df.loc[ii, "year"] = outlier_df.loc[ii, "Datum"].year

perc = 1
lower_threshold = outlier_df.groupby("year")["Preis"].quantile(q=perc / 100)
upper_threshold = outlier_df.groupby("year")["Preis"].quantile(q=1 - perc / 100)

outlier_df = outlier_df.merge(
    right=lower_threshold, on="year", suffixes=("", "_lower")
)
outlier_df = outlier_df.merge(
    right=upper_threshold, on="year", suffixes=("", "_upper")
)
outlier_df.head()
outlier_df.columns

fig_outliers = (
    ggplot(outlier_df)
    + geom_line(mapping=aes("Datum", "Preis"))
    + geom_line(mapping=aes("Datum", "Preis_lower"), color="red")
    + geom_line(mapping=aes("Datum", "Preis_upper"), color="red")
    + ggtitle("Hourly Prices: Detecting Outliers")
    + labs(x="Time", y="Prices")
)
fig_outliers.draw(show=True)
fig_outliers.save("hi.pdf")


# === Remove Outliers ===
down_spike = outlier_df["Preis"] < outlier_df["Preis_lower"]
up_spike = outlier_df["Preis"] > outlier_df["Preis_upper"]

wo_outlier_df = outlier_df.copy()

sel = down_spike + up_spike

# wo_outlier_df.loc[down_spike, "Preis"] = np.nan
# wo_outlier_df.loc[up_spike, "Preis"] = np.nan

smoothed_preis = (
    wo_outlier_df.loc[:, "Preis"]
    .rolling(window=168, min_periods=1, center=True)
    .mean()
)

wo_outlier_df.loc[sel, "Preis"] = smoothed_preis[sel]

# wo_outlier_df.interpolate(method="time", inplace=True)

fig_wo_outliers = (
    ggplot(wo_outlier_df, mapping=aes("Datum", "Preis"))
    + geom_line()
    + ggtitle("Hourly Prices: After Removing Outliers")
    + labs(x="Time", y="Prices")
)
fig_wo_outliers.draw(show=True)
fig_wo_outliers.save("hi2.pdf")

his_data = wo_outlier_df[["Datum", "Preis"]]
his_data.head()


# === Hourly, Daily and Yearly Prices ===
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


# === Normalize the prices ===
d_df = d_df.merge(right=y_df[["year", "y_preis"]], on="year")

d_df["nd_preis"] = d_df["d_preis"] / d_df["y_preis"]

d_df.head()


h_df = h_df.merge(
    right=d_df[["year", "month", "day", "nd_preis"]],
    on=["year", "month", "day"],
)

h_df["nh_preis"] = h_df["h_preis"] / h_df["nd_preis"]

h_df.head()


# === Save the data sets ===
h_df.to_csv("h_df.csv")
d_df.to_csv("d_df.csv")


# === Plot Hourly Prices ===
# Hourly price
fig_h_preis = (
    ggplot(h_df, mapping=aes("Datum", "h_preis"))
    + geom_line()
    # + geom_hline(yintercept=0)
    # + facet_wrap("year", scales="free")
    + ggtitle("Hourly prices")
    + labs(x="Time", y="Prices")
)

fig_h_preis.draw(show=True)


# Normalized hourly prices
fig_nh_preis = (
    ggplot(h_df, mapping=aes(x="Datum", y="nh_preis"))
    + geom_line()
    # + facet_wrap("year", scales="free")
    + ggtitle("Normalized hourly prices")
    + labs(x="Time", y="Prices")
)

fig_nh_preis.draw(show=True)


# === Plot Daily Prices ===
fig_d_preis = (
    ggplot(d_df, mapping=aes(x="date", y="d_preis"))
    + geom_line()
    # + facet_wrap("year", scales="free")
    + ggtitle("Daily prices")
    + labs(x="Time", y="Prices")
)

fig_d_preis.draw(show=True)


# Normalized daily prices
fig_nd_preis = (
    ggplot(d_df, mapping=aes(x="date", y="nd_preis"))
    + geom_line()
    # + facet_wrap("year", scales="free")
    + ggtitle("Normalized daily prices")
    + labs(x="Time", y="Prices")
)

fig_nd_preis.draw(show=True)


# === Plot Yearly Prices ===
fig_y_preis = (
    ggplot(y_df, mapping=aes(x="year", y="y_preis"))
    + geom_line()
    + ggtitle("Yearly Prices")
    + labs(x="Year", y="Price")
)
fig_y_preis.draw(show=True)


# === Box Plots -- Monthly ===
fig = (
    ggplot(d_df, mapping=aes(x="factor(month)", y="nd_preis"))
    + geom_boxplot()
    + ggtitle("Daily prices")
    + labs(x="Time", y="Prices")
)
fig.draw(show=True)


# === Box Plots -- Daily ===


# === Autocorrelation Plots ===
