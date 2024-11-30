import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Loading data ===
his_data = pd.read_excel(io="2024 Historische day-ahead-Preise Strom DE.xlsx")

his_data.head()

his_data.shape


# === Transformations ===
# Split the date
his_data.insert(loc=1, column="hour", value=np.nan)
his_data.insert(loc=1, column="day", value=np.nan)
his_data.insert(loc=1, column="month", value=np.nan)
his_data.insert(loc=1, column="year", value=np.nan)

for ii in range(his_data.shape[0]):
    his_data.loc[ii, "hour"] = his_data.loc[ii, "Datum"].hour
    his_data.loc[ii, "day"] = his_data.loc[ii, "Datum"].day
    his_data.loc[ii, "month"] = his_data.loc[ii, "Datum"].month
    his_data.loc[ii, "year"] = his_data.loc[ii, "Datum"].year

his_data.head()

# Hourly prices
h_df = his_data

h_df.head()

# Daily prices
d_df = h_df.groupby(["year", "month", "day"], as_index=False).mean()

d_df = d_df.loc[:, ("year", "month", "day", "Preis")]

# TODO: make date column at loc 0

d_df.head()

# Yearly prices
y_df = d_df.groupby("year", as_index=False).mean()

y_df = y_df.loc[:, ("year", "Preis")]

# TODO: make date column at loc 0

y_df.head()

# Calculate the averages
h_df.merge(right=d_df, on=["year", "month", "day"])

h_df.head()

# Get weekdays
his_data.insert(loc=0, column="weekday", value=np.nan)
for ii in range(his_data.shape[0]):
    his_data.loc[ii, "weekday"] = his_data.loc[ii, "Datum"].weekday()

# Normalize the prices

# Build the clusters


# === Data exploration
plt.plot(his_data["Preis"])
plt.show()


# === Regressions ===
