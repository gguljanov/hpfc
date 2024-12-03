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


# === Seasonality dummies & Regression for a year ===
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


X_mat_day = d_df_dummies.loc[:, "weekday_1":"month_12"]
X_mat_day.head()
X_mat_day.tail()

Y_vec_day = d_df["nd_preis"]
Y_vec_day.head()
Y_vec_day.tail()

reg_year_shape = LinearRegression().fit(X=X_mat_day, y=Y_vec_day)

reg_year_shape.score(X=X_mat_day, y=Y_vec_day)


# === Seasonality dummies & Regression for a day ===
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

X_mat_hour = h_df_dummies.loc[:, "hour_1":"sun_Nov"]
Y_vec_hour = h_df_dummies.loc[:, "nh_preis"]

reg_day_shape = LinearRegression().fit(X=X_mat_hour, y=Y_vec_hour)
reg_day_shape.score(X=X_mat_hour, y=Y_vec_hour)


# === Actual vs. Predicted values ===
nd_pred = reg_year_shape.predict(X_mat_day)
d_df["nd_pred"] = nd_pred

nh_pred = reg_day_shape.predict(X_mat_hour)
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


# === Autocorrelation before and after de-seasonalization ===
# To be done


# === MAE, RMSE ===
rprint(mean_absolute_error(y_true=h_df["h_preis"], y_pred=h_df["h_pred"]))
rprint(mean_squared_error(y_true=h_df["h_preis"], y_pred=h_df["h_pred"]))


# === Backtesting ===
selvec = d_df["year"] == 2023
np.sum(selvec)

X_mat_day_train = X_mat_day.loc[np.logical_not(selvec), :]
Y_vec_day_train = Y_vec_day.loc[np.logical_not(selvec)]
X_mat_day_test = X_mat_day.loc[selvec, :]
Y_vec_day_test = Y_vec_day.loc[selvec]

selvec_h = h_df["year"] == 2023
X_mat_hour_train = X_mat_hour.loc[np.logical_not(selvec_h), :]
Y_vec_hour_train = Y_vec_hour.loc[np.logical_not(selvec_h)]
X_mat_hour_test = X_mat_hour.loc[selvec_h, :]
Y_vec_hour_test = Y_vec_hour.loc[selvec_h]

reg_day = LinearRegression().fit(X=X_mat_day_train, y=Y_vec_day_train)
reg_hour = LinearRegression().fit(X=X_mat_hour_train, y=Y_vec_hour_train)

day_pred = reg_day.predict(X_mat_day_test)
hour_pred = reg_hour.predict(X_mat_hour_test)

d_df_test = d_df.loc[selvec, :]
h_df_test = h_df.loc[selvec_h, :]

d_df_test.loc[:, "nd_pred_out"] = day_pred
h_df_test.loc[:, "nh_pred_out"] = hour_pred

h_df_test = h_df_test.merge(
    right=d_df_test[["year", "month", "day", "nd_pred_out"]],
    on=["year", "month", "day"],
)

rprint(h_df_test.columns)

h_df_test["h_pred_out"] = h_df_test["nh_pred_out"] * h_df_test["nd_pred_out"]
rprint(h_df.columns)

fig_h_pred_out = (
    # ggplot(
    #     h_df_test[h_df_test["month"] == 1], mapping=aes(x="Datum", y="h_preis")
    # )
    ggplot(h_df_test, mapping=aes(x="Datum", y="h_preis"))
    + geom_line()
    + geom_line(mapping=aes(y="h_pred_out"), color="yellow")
    + facet_wrap("year", scales="free")
    + ggtitle("Actual and Predicted Hourly Prices")
    + labs(x="Time", y="Prices")
)

fig_h_pred_out.draw(show=True)

rprint(
    mean_absolute_error(y_true=Y_vec_hour_test, y_pred=h_df_test["h_pred_out"])
)
rprint(mean_squared_error(y_true=Y_vec_hour_test, y_pred=hour_pred))


# === EEX ===
# To be done
