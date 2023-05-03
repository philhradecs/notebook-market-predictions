import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sphist = pd.read_csv('sphist.csv', parse_dates=['Date'])
sphist = sphist.sort_values('Date').reset_index(drop=True)
original_cols = set(sphist.columns)

def rolling_mean(column, windows, add_columns=False):
    results = []
    for window in windows:
        name = column + '_day_' + str(window)
        data = sphist[column].rolling(window).mean().shift(1) 
        if add_columns:
            sphist[name] = data 
        results.append((data, name))
        
rolling_mean('Close', [5, 30, 365], add_columns=True)
rolling_mean('Volume', [5, 365], add_columns=True)

sphist['Volume_avg_std_day_5'] = sphist['Volume'].rolling(5).std().shift(1)

sphist['weekday'] = sphist['Date'].dt.weekday

sphist = sphist[sphist['Date'] >= '1951-01-03']
sphist = sphist.dropna(axis=0)

def train_test(train, test):
    if len(train) == 0:
        return np.nan
    
    feature_cols = list(set(train.columns) - original_cols)
    target_col = 'Close'

    lr = LinearRegression()
    lr.fit(train[feature_cols], train[target_col])
    predictions = lr.predict(test[feature_cols])

    mae = mean_absolute_error(test[target_col], predictions)
    return mae

print('fixed TRAIN / TEST split')
train = sphist[sphist['Date'] < '2013-01-01']
test = sphist[sphist['Date'] >= '2013-01-01']
msa = train_test(train, test)
print('MAE', msa)

print('predicting one day at a time')
predict_dates = sphist['Date'][sphist['Date'] >= '2013-01-01']

msa_values = []
rmse_values = []


mae_values = [ 
    train_test(sphist[sphist['Date'] < date], sphist[sphist['Date'] == date])
    for date in predict_dates
]

print('avg MAE', np.mean(mae_values))
