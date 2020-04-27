import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#load dữ liệu
data = pd.read_csv("D:\AHocMay\CodeThucHanh\data.csv", encoding='utf-8', sep=';')
one = np.ones((data.shape[0], 1))
data.insert(loc=0, column='A', value=one) #add 1 in each row of column A
data_x = data[["A", "Height"]]
data_y = data["Weight"]
#Tách training và test sets
X_train, X_test, y_train, y_test = train_test_split(
     data_x, data_y, test_size=5)
# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(X_train, y_train)
Y_pred = regr.predict(X_test) 
#plt.scatter(data.Height, data.Weight)
plt.plot(data.Height, data.Weight, 'ro-')
#plt.plot(data.Height, data.Weight, color='black')
plt.plot(X_test.Height, Y_pred, color='blue')
w_0 = regr.coef_[0]
w_1 = regr.coef_[1]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0
plt.plot(x0, y0, color ='pink')
plt.show()

#predict
#example: predict with x0=26
y1=w_0 + w_1*26
print(y1)