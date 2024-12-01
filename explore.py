import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from plotnine.ggplot import ggplot
from plotnine import (
    aes,
    labs,
    facet_wrap,
    geom_point,
    geom_hline,
    geom_line,
    geom_boxplot,
    ggtitle,
    xlab,
    ylab,
)


# === Loading data ===
his_data = pd.read_excel(io="2024 Historische day-ahead-Preise Strom DE.xlsx")

his_data.head()

his_data.shape


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
h_df = his_data

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


# === Regressions ===
hi = pd.get_dummies(d_df, prefix="month", drop_first=True, columns=["month"])
pd.concat([d_df, hi], axis=1)

# First tests
# Download from eex
# Check on the eex data


# === MAE, RMSE ===


# === Backtesting ===
