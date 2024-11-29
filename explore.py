import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

his_data = pd.read_excel(io="2024 Historische day-ahead-Preise Strom DE.xlsx")

his_data.head()

his_data.shape

his_data.insert(loc=0, column="year", value=np.nan)
his_data.insert(loc=0, column="month", value=np.nan)
his_data.insert(loc=0, column="day", value=np.nan)
his_data.insert(loc=0, column="hour", value=np.nan)
his_data.insert(loc=0, column="weekday", value=np.nan)

for ii in range(his_data.shape[0]):
    his_data.loc[ii, "year"] = his_data.loc[ii, "Datum"].year
    his_data.loc[ii, "month"] = his_data.loc[ii, "Datum"].month
    his_data.loc[ii, "day"] = his_data.loc[ii, "Datum"].day
    his_data.loc[ii, "hour"] = his_data.loc[ii, "Datum"].hour
    his_data.loc[ii, "weekday"] = his_data.loc[ii, "Datum"].weekday()

his_data.head()

# Daily averages
new_df = his_data.groupby(["year", "month", "day"], as_index=False).mean()

his_data.merge(right=new_df, left_on="")


plt.plot(his_data["Preis"])
plt.show()
