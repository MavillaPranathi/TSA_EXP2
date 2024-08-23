# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### Developed by : M.Pranathi
### Register no : 212222240064
### Date:

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program

### PROGRAM:
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv('supermarketsales.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data['Date_Ordinal'] = data['Date'].apply(lambda x: x.toordinal())
X = data['Date_Ordinal'].values.reshape(-1, 1)  
y = data['Total'].values 

A - LINEAR TREND ESTIMATION

linear_model = LinearRegression()
linear_model.fit(X, y)
data['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Total'],label='Original Data')  # 'bo-' for blue circles connected with a line
plt.plot(data['Date'], data['Linear_Trend'], color='red', label='Linear Trend')  # 'r-' for a red line
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Total')
plt.legend()
plt.grid(True)
plt.show()

B- POLYNOMIAL TREND ESTIMATION

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
data['Polynomial_Trend'] = poly_model.predict(X_poly)
plt.figure(figsize=(10,6))
plt.bar(data['Date'], data['Total'], label='Original Data', alpha=0.6)  # Bar plot with transparency
plt.plot(data['Date'], data['Polynomial_Trend'],color='red', label='Polynomial Trend (Degree 2)')  # 'g-' for a green line
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Total')
plt.legend()
plt.grid(True)
plt.show()

```
### OUTPUT

A - LINEAR TREND ESTIMATION


![image](https://github.com/user-attachments/assets/6b94c659-035a-4823-a7ef-0cb4483a2afc)


B- POLYNOMIAL TREND ESTIMATION

![image](https://github.com/user-attachments/assets/c966eca9-25c0-4752-bd7e-fdf7910909c1)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
