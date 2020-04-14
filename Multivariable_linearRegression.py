import numpy as np
#import tensorflow as tf #cant find the tensorflow package
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#create some test data and simulate results
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real,x_data.T) + b_real + noise

print(len(x_data))
print(len(y_data[0]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = x_data[:,0]
x2 = x_data[:,1]
x3 = x_data[:,2]
ax.scatter3D(x1, x2, y_data[0], c=y_data[0], cmap='plasma');

plt.show()

#actual implementation of liner regression
#compute y_pred, compare with y_data above etc etc
#assume more code here

exit() 