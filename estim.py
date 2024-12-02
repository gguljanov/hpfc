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


# === Load data ===
h_df = pd.read_csv("h_df.csv")
h_df.head()

d_df = pd.read_csv("d_df.csv", date_format="yyyy-mm-dd", index_col=0)
d_df.head()


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
