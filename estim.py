from rich import print as rprint

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


# === Load data ===
h_df = pd.read_csv("h_df.csv", index_col=0, parse_dates=True)
h_df.head()

h_df["Datum"] = pd.to_datetime(h_df["Datum"])

d_df = pd.read_csv("d_df.csv", date_format="yyyy-mm-dd", index_col=0)
d_df.head()

d_df["date"] = pd.to_datetime(d_df["date"])


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

nd_pred = reg_year_shape.predict(X_mat)
d_df["nd_pred"] = nd_pred

nh_pred = reg_day_shape.predict(X_mat_day)
h_df["nh_pred"] = nh_pred

h_df = h_df.merge(
    right=d_df[["year", "month", "day", "nd_pred"]], on=["year", "month", "day"]
)

rprint(h_df.head())

h_df["h_pred"] = h_df["nh_pred"] * h_df["nd_pred"]
rprint(h_df.head())

fig_h_pred = (
    ggplot(h_df[h_df["month"] == 1], mapping=aes(x="Datum", y="h_preis"))
    + geom_line()
    + geom_line(mapping=aes(y="h_pred"), color="yellow")
    + facet_wrap("year", scales="free")
    + ggtitle("Actual and Predicted Hourly Prices")
    + labs(x="Time", y="Prices")
)

fig_h_pred.draw(show=True)


# === Autocorrelation before and after de-seasonalization


# === MAE, RMSE ===
rprint(mean_absolute_error(y_true=h_df["h_preis"], y_pred=h_df["h_pred"]))
rprint(mean_squared_error(y_true=h_df["h_preis"], y_pred=h_df["h_pred"]))


# === Backtesting ===


# Download from eex
# Check on the eex data
