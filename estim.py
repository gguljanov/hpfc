from rich import print as rprint

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

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
    lims,
)


# === Load data ===
h_df = pd.read_csv("h_df.csv", index_col=0, parse_dates=True)
h_df.head()

h_df["Datum"] = pd.to_datetime(h_df["Datum"])

d_df = pd.read_csv("d_df.csv", date_format="yyyy-mm-dd", index_col=0)
d_df.head()

d_df["date"] = pd.to_datetime(d_df["date"])

y_df = pd.read_csv("y_df.csv", index_col=0)


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

for ii in range(1, 12):
    month_name = month_names[ii - 1]

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


# === Box Plots for weekday, Sat, and Sun in months ===
wd_names = ["wd_" + xx for xx in month_names]
sat_names = ["sat_" + xx for xx in month_names]
sun_names = ["sun_" + xx for xx in month_names]

h_df_dummies["wd_month"] = pd.from_dummies(
    h_df_dummies.loc[:, wd_names], default_category="wd_Dec"
)

h_df_dummies["sat_month"] = pd.from_dummies(
    h_df_dummies.loc[:, sat_names], default_category="sat_Dec"
)

h_df_dummies["sun_month"] = pd.from_dummies(
    h_df_dummies.loc[:, sun_names], default_category="sun_Dec"
)

cats = wd_names.copy()
cats.append("wd_Dec")

# Box Plots -- wd_month
fig_box_wd_month = (
    ggplot(
        h_df_dummies,
        mapping=aes(x="factor(wd_month, categories=cats)", y="nh_preis"),
    )
    + geom_boxplot()
    + ggtitle("Weekday in Each Month")
    + labs(x="Time", y="Prices")
    + lims(y=[0.75, 1.25])
)

# fig_box_wd_month.draw(show=True)
fig_box_wd_month.save(filename="fig_box_wd_month.pdf", path="Figs")

cats_sat = sat_names.copy()
cats_sat.append("sat_Dec")


# Box Plots -- sat_month
fig_box_sat_month = (
    ggplot(
        h_df_dummies,
        mapping=aes(x="factor(sat_month, categories=cats_sat)", y="nh_preis"),
    )
    + geom_boxplot()
    + ggtitle("Saturday in Each Month")
    + labs(x="Time", y="Prices")
    + lims(y=[0.75, 1.25])
)

# fig_box_sat_month.draw(show=True)
fig_box_sat_month.save(filename="fig_box_sat_month.pdf", path="Figs")


cats_sun = sun_names.copy()
cats_sun.append("sun_Dec")


# Box Plots -- sun_month
fig_box_sun_month = (
    ggplot(
        h_df_dummies,
        mapping=aes(x="factor(sun_month, categories=cats_sun)", y="nh_preis"),
    )
    + geom_boxplot()
    + ggtitle("Sunday in Each Month")
    + labs(x="Time", y="Prices")
    + lims(y=[0.75, 1.25])
)

# fig_box_sun_month.draw(show=True)
fig_box_sun_month.save(filename="fig_box_sun_month.pdf", path="Figs")


# === Actual vs. Predicted values ===
nd_pred = reg_year_shape.predict(X_mat_day)
d_df["nd_pred"] = nd_pred

nh_pred = reg_day_shape.predict(X_mat_hour)
h_df["nh_pred"] = nh_pred

h_df = h_df.merge(
    right=d_df[["year", "month", "day", "nd_pred"]], on=["year", "month", "day"]
)

h_df = h_df.merge(right=y_df, on=["year"])

rprint(h_df.head())

h_df["h_pred"] = h_df["nh_pred"] * h_df["nd_pred"] * h_df["y_preis"]
rprint(h_df.head())

fig_h_pred = (
    # ggplot(h_df[h_df["month"] == 1], mapping=aes(x="Datum", y="h_preis"))
    ggplot(h_df, mapping=aes(x="Datum", y="h_preis"))
    + geom_line()
    + geom_line(mapping=aes(y="h_pred"), color="yellow")
    + facet_wrap("year", scales="free")
    + ggtitle("Actual and Predicted Hourly Prices")
    + labs(x="Time", y="Prices")
)

# fig_h_pred.draw(show=True)
fig_h_pred.save(filename="fig_h_pred.pdf", path="Figs")


# === Autocorrelation before and after de-seasonalization ===
plot_acf(d_df["nd_preis"])
# plt.show()
plt.savefig("Figs/daily_prices-before_seas.pdf")

res_daily = d_df["nd_preis"] - d_df["nd_pred"]
plot_acf(res_daily)
# plt.show()
plt.savefig("Figs/daily_prices-after_seas.pdf")

plot_acf(h_df["nh_preis"])
# plt.show()
plt.savefig("Figs/hourly_prices-before_seas.pdf")

res_hourly = h_df["nh_preis"] - h_df["nh_pred"]
plot_acf(res_hourly)
# plt.show()
plt.savefig("Figs/hourly_prices-after_seas.pdf")


# === MAE, RMSE ===
fit_df = pd.DataFrame(
    index=["in-sample", "out-of-sample"], columns=["MAE", "RMSE"]
)

fit_df.loc["in-sample", "MAE"] = mean_absolute_error(
    y_true=h_df["h_preis"], y_pred=h_df["h_pred"]
)
fit_df.loc["in-sample", "RMSE"] = mean_squared_error(
    y_true=h_df["h_preis"], y_pred=h_df["h_pred"]
)

rprint(fit_df)


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

h_df_test["h_pred_out"] = (
    h_df_test["nh_pred_out"] * h_df_test["nd_pred_out"] * h_df_test["y_preis"]
)
rprint(h_df.columns)

fig_h_pred_out = (
    ggplot(
        h_df_test[h_df_test["month"] == 1], mapping=aes(x="Datum", y="h_preis")
    )
    # ggplot(h_df_test, mapping=aes(x="Datum", y="h_preis"))
    + geom_line()
    + geom_line(mapping=aes(y="h_pred_out"), color="yellow")
    + facet_wrap("year", scales="free")
    + ggtitle("Actual and Predicted Hourly Prices, Out-of-Sample")
    + labs(x="Time", y="Prices")
)

fig_h_pred_out.draw(show=True)
fig_h_pred_out.save(filename="fig_h_pred_out.pdf", path="Figs")

# Out-of-sample fit
fit_df.loc["out-of-sample", "MAE"] = mean_absolute_error(
    y_true=Y_vec_hour_test, y_pred=h_df_test["h_pred_out"]
)
fit_df.loc["out-of-sample", "RMSE"] = mean_squared_error(
    y_true=Y_vec_hour_test, y_pred=h_df_test["h_pred_out"]
)

rprint(fit_df)

fit_df.to_latex(buf="Tables/fit_df.tex")

# === EEX ===
# To be done
