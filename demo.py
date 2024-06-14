import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data = pd.read_csv('weather_data.csv')

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Set the date column as the index
data.set_index('date', inplace=True)

# Drop any rows with missing values
data.dropna(inplace=True)

# Feature selection: let's use previous day temperature to predict the next day temperature
data['previous_temp'] = data['temperature'].shift(1)

# Drop the first row as it will have NaN value for previous_temp
data.dropna(inplace=True)

# Define features and target variable
X = data[['previous_temp']]
y = data['temperature']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the true values vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()

# Save the model to a file
joblib.dump(model, 'weather_prediction_model.pkl')
