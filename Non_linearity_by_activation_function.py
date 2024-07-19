import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generating synthetic data: house sizes (input) and prices (output)
np.random.seed(42)
house_sizes = np.random.uniform(500, 3500, 1000)
noise = np.random.normal(0, 50000, 1000)
house_prices = (30000 + house_sizes * 150 + noise)

# Splitting the data into training and testing sets
train_sizes = house_sizes[:800]
train_prices = house_prices[:800]
test_sizes = house_sizes[800:]
test_prices = house_prices[800:]

# Building the neural network model
model = Sequential([
    Dense(32, input_dim=1, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')

# Training the model
history = model.fit(train_sizes, train_prices, epochs=100, validation_split=0.2, verbose=0)

# Predicting house prices
predicted_prices = model.predict(test_sizes)

# Plotting the results
plt.scatter(test_sizes, test_prices, color='blue', label='Actual Prices')
plt.scatter(test_sizes, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel('House Size (sqft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()
