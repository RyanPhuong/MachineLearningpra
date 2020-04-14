import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#read data
df = pd.read_csv("winequality-red.csv", encoding="utf-8", sep =";")
correlations = df.corr()["quality"].drop("quality")
print(correlations)
#sns.heatmap(df.corr())#get detailed diagram of correlations
#plt.show()
#define a func which output those feature whose correlations
def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations
features = get_features(0.05) 
print(features) 
x = df[features]
y = df['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state =3)
#create training test set using train_test_split with 75% data for training

#fitting linear regression to training data
regressor = LinearRegression()
regressor.fit(x_train, y_train)
#this give the coefficients of the 10 features selected above
print(regressor.coef_)

train_pred = regressor.predict(x_train)
print(train_pred)
test_pred = regressor.predict(x_test) 
print(test_pred)
# calculating rmse
train_rmse = mean_squared_error(train_pred, y_train) ** 0.5
print(train_rmse)
test_rmse = mean_squared_error(test_pred, y_test) ** 0.5
print(test_rmse)
# rounding off the predicted values for test set
predicted_data = np.round_(test_pred)
print(predicted_data)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))
# displaying coefficients of each feature
coeffecients = pd.DataFrame(regressor.coef_,features) #coeffecients.columns = ['Coeffecient'] 
print(coeffecients)
