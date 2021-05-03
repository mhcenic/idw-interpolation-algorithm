#!/usr/bin/env python
# coding: utf-8

# # Data pre-processing

# In[117]:


import glob
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error


# In[118]:


ORIGINAL_DIR = "../Original-Data"
ANALYSIS_DIR = "../Analysis-Data"


# In[119]:


all_files = glob.glob(os.path.join(ORIGINAL_DIR,"*2017.csv")) 
all_data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)


# In[120]:


all_data.head()


# In[121]:


pm10_df = all_data.filter(regex='UTC time|pm10')
pm10_df.shape


# In[122]:


pm10_df.head()


# In[123]:


#According to 'Tidy data' rules
pm10_df_with_id = pd.DataFrame(columns=['UTC time', 'pm10', 'id'])
for col_name in pm10_df.iloc[:, 1:]:
    df_with_id = pd.DataFrame(columns=['UTC time', 'pm10', 'id'])
    df_with_id['UTC time'] = pm10_df['UTC time']
    df_with_id['pm10'] = pm10_df[col_name]
    df_with_id['id'] = col_name
    pm10_df_with_id = pm10_df_with_id.append(df_with_id)


# In[124]:


pm10_df_with_id.shape


# In[125]:


pm10_df_with_id.head()


# In[126]:


pm10_df_with_id.id = pm10_df_with_id.id.str.replace('_pm10','').astype(int)


# In[127]:


pm10_df_with_id.head()


# In[128]:


sensors = pd.read_csv(f"{ORIGINAL_DIR}/sensor_locations.csv")
sensors.head()


# In[129]:


cleaned_data = (pm10_df_with_id.merge(sensors, left_on='id', right_on='id')
       .reindex(columns=['UTC time', 'pm10', 'id', 'latitude', 'longitude']))


# In[130]:


cleaned_data.head()


# In[131]:


#Remove unnecessary columns
cleaned_data.drop('id', inplace=True, axis=1)


# In[132]:


cleaned_data.head()


# In[133]:


cleaned_data['pm10'].isnull().sum()


# In[134]:


#Remove missing rows
cleaned_data.dropna(subset = ["pm10"], inplace=True)


# In[135]:


cleaned_data.rename(columns={'UTC time': 'datetime'}, inplace=True)
cleaned_data.datetime = pd.to_datetime(cleaned_data.datetime)
cleaned_data.head()


# In[136]:


cleaned_data.shape


# In[137]:


cleaned_data['datetime'].value_counts()


# # Algorithm

# In[138]:


from photutils.utils import ShepardIDWInterpolator as idw
from sklearn.model_selection import train_test_split
from typing import NewType
import matplotlib.pyplot as plt


# In[139]:


def get_data(dataframe, time):
    return dataframe.loc[dataframe.datetime == time]


# **Hyperparameters tuning**

# The Shepard interpolator has an available number of parameters that are listed in the documentation.
# 
# https://photutils.readthedocs.io/en/stable/api/photutils.utils.ShepardIDWInterpolator.html
# 
# **We will adjust the following parameters:**
# 
# _n\_neighbors_ -> The maximum number of nearest neighbors to use during the interpolation
# 
# _power_ -> The power of the inverse distance used for the interpolation weights.

# I chose _2017-03-17 13:00:00_  because it contains the most data (50 samples)

# In[140]:


MAX_NEIGHBORS = 15
POWER_RANGE = np.arange(0.1, 5.0, 0.2)


# In[141]:


selected = get_data(cleaned_data, '2017-03-17 13:00:00')


# In[142]:


X_train, X_test, y_train, y_test = train_test_split(selected[["latitude","longitude"]].values, 
                                                    selected["pm10"].values, test_size=0.2, random_state=42)


# In[143]:


# Run idw interpolator
f = idw(X_train, y_train)


# In[144]:


for n_neighbors in range(3, MAX_NEIGHBORS):
    power_list=[]
    rmse_list=[]
    for power in POWER_RANGE:
        predictions = f(X_test, n_neighbors = n_neighbors, power = power)
        rmse = mean_squared_error(y_test, predictions, squared = False)
        power_list.append(power)
        rmse_list.append(rmse)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.plot(power_list, rmse_list, label=n_neighbors)
plt.grid()
plt.xlabel('power')
plt.ylabel('RMSE')
plt.legend(title="n_neighbors")
plt.show()


# We can notice that the model achieves satisfactory results for n_neighbors equal to 4 and the model is not very complex.

# _We will now check which value of the power parameter will produce the best results for n_neihbors equal to 4._

# In[145]:


for n_neighbors in range(3, MAX_NEIGHBORS):
    power_list=[]
    rmse_list=[]
    print(f"n_neighbors: {n_neighbors}")
    print(f"power \t\t rmse")
    for power in POWER_RANGE:
        predictions = f(X_test, n_neighbors = n_neighbors, power = power)
        rmse = mean_squared_error(y_test, predictions, squared = False)
        power_list.append(power)
        rmse_list.append(rmse)
        print(f"{power:.1f} \t\t {rmse:.4f}")
    plt.rcParams["figure.figsize"] = (5,5)
    plt.plot(power_list, rmse_list, label=n_neighbors)
    plt.grid()
    plt.xlabel('power')
    plt.ylabel('RMSE')
    plt.legend(title="n_neighbors")
    plt.show()


# **Summing up the best interpolation result was obtained for the n-neighbors parameters equal to 4 and for the variable power equal to 0.1.**
