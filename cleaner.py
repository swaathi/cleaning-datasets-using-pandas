import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('data.csv')

df.drop(['item'], axis=1, inplace=True)
df.drop(['importer_id'], axis=1, inplace=True)
df.drop(['exporter_id'], axis=1, inplace=True)
df.drop(['mode_of_transport'], axis=1, inplace=True)
df.drop(['days_in_transit'], axis=1, inplace=True)

dummies = pd.get_dummies(df['route'], prefix='route')
df = pd.concat([df, dummies], axis=1)
df.drop(['route'], axis=1, inplace=True)

dummies = pd.get_dummies(df['country_of_origin'], prefix='origin')
df = pd.concat([df, dummies], axis=1)
df.drop(['country_of_origin'], axis=1, inplace=True)

df['declared_quantity'] = minmax_scaler.fit_transform(df['declared_quantity'].values.reshape(-1,1))
df['declared_cost'] = minmax_scaler.fit_transform(df['declared_cost'].values.reshape(-1,1))

df['weight_diff'] = (
    (df['actual_weight']-df['declared_weight']) / df['actual_weight']
).round(3)*100
df['weight_diff'] = minmax_scaler.fit_transform(df['weight_diff'].values.reshape(-1,1))

df.drop(['declared_weight'], axis=1, inplace=True)
df.drop(['actual_weight'], axis=1, inplace=True)

df['date_of_arrival'] = pd.to_datetime(df['date_of_arrival'])
df['date_of_departure'] = pd.to_datetime(df['date_of_departure'])

df['days_diff'] = df['date_of_arrival'] - df['date_of_departure']
df['days_diff'] = df['days_diff'] / np.timedelta64(1,'D')
df['days_diff'] = minmax_scaler.fit_transform(df['days_diff'].values.reshape(-1,1))

df.drop(['date_of_arrival'], axis=1, inplace=True)
df.drop(['date_of_departure'], axis=1, inplace=True)

X = df.drop(['valid_import'], axis=1)
y = df['valid_import']

print(X.head())
