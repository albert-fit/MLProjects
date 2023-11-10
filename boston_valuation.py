# from sklearn.datasets import load_boston /////deprecated\
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
load_boston = fetch_openml(data_id=531, as_frame=True, parser='pandas')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data

boston_dataset = load_boston
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)
log_prices = np.log(boston_dataset.target)

target = log_prices.to_frame(name=['PRICES'])
data.head()


property_stats = np.ndarray(shape=(1,11))

PTRATIO_IDX = 8
RM_IDX = 4
CHAS_IDX = 3
ZILLOW_MEDIAN_PRICE = 584.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

property_stats = features.astype(float).mean().values.reshape(1,11)


regr = LinearRegression().fit(features.astype(float), target)
fitted_vals = regr.predict(features.astype(float))

#Challange: calculate the MSE and RMSE using sklearn

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms, 
                     students_per_classroom,
                     next_to_river=False,
                     high_confidence=True) : 
    #Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    

    #Make prediction
    log_estimate = regr.predict(property_stats)[0][0]

    #Calc Range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68


    return log_estimate, upper_bound, lower_bound, interval

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):

    """
    Estimate the price of a property in Boston.\n

    Keywords:\n
        rm -- Number of rooms;\n
        ptratio -- Number of pupils per teacher in the classroom for the school in the area; \n
        chas -- True if the property is next to the river;\n
        large_range -- True for a 95% prediction interval, False for a 68% prediction interval\n

    """

    if rm < 1 or ptratio < 1:
        print('That is unrealistic. Try again!')
        return
    

    log_est, uppper, lower, conf = get_log_estimate(rm, students_per_classroom = ptratio, next_to_river = chas, high_confidence = large_range)
    
    #Convert to today's dollars
    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi = np.e**uppper * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower * 1000 * SCALE_FACTOR
    dollar_est

    #Round the dollar values to nearest 1000
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)
    rounded_est

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {conf}% confidence the valuation range is')
    print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end.')